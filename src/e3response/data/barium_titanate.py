import functools
import logging
import pathlib
import re
import tarfile
from typing import Any, Callable, Final, Iterable, Optional, Sequence, Union

import ase
import ase.io
from ase.io import espresso
import jraph
import numpy as np
import reax
from tensorial import gcnn  # pylint: disable=no-name-in-module
from typing_extensions import override

__all__ = ("BtoDataModule",)

_LOGGER = logging.getLogger(__name__)


class BtoDataModule(reax.DataModule):
    """A barium titanate dataset containing various tensorial quantities created by Lorenzo
    Bastonero"""

    _max_padding: gcnn.data.GraphPadding = None

    def __init__(
        self,
        r_max: float,
        data_dir: Union[str, pathlib.Path] = "data/bto/",
        archives: Sequence[str] = (
            "BTO_Pm-3m_5atoms_400K_3x3x3_ensemble.tar.gz",
            "BTO_Pm-3m_5atoms_800K_3x3x3.tar.gz",
        ),
        tensors: tuple[str] = ("raman_tensors", "born_charges"),
        train_val_test_split: Sequence[Union[int, float]] = (0.85, 0.05, 0.1),
        batch_size: int = 64,
    ) -> None:
        """Initialize a `SiliconDataModule`.

        :param traj_file: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split.
        :param batch_size: The batch size. Defaults to `64`.
        """
        super().__init__()

        # Params
        self._rmax = r_max
        self._data_dir: Final[str] = str(data_dir)
        self._archives: Final[tuple[str, ...]] = tuple(archives)
        self._tensors = tensors
        self._train_val_test_split: Final[Sequence[Union[int, float]]] = train_val_test_split
        self._batch_size: Final[int] = batch_size

        # State
        self.batch_size_per_device = batch_size
        self.data_train: Optional[reax.data.Dataset] = None
        self.data_val: Optional[reax.data.Dataset] = None
        self.data_test: Optional[reax.data.Dataset] = None

    @override
    def setup(self, stage: "reax.Stage", /) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by REAX before `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`, so be careful not to execute things like random
        split twice! Also, it is called after `self.prepare_data()` and there is a barrier in
        between which ensures that all the processes proceed to `self.setup()` once the data is
        prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        Defaults to ``None``.
        """
        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            all_train = []
            all_val = []
            all_test = []

            for archive in self._archives:
                output_dir = self._extract_tarball(archive)

                structures = get_structures(output_dir, self._tensors)

                # Split up the graphs into sets
                train, val, test = reax.data.random_split(
                    stage.rng, dataset=structures, lengths=self._train_val_test_split
                )
                all_train.extend(train)
                all_val.extend(val)
                all_test.extend(test)

            to_graph: Callable[[ase.Atoms], jraph.GraphsTuple] = functools.partial(
                gcnn.atomic.graph_from_ase,
                r_max=self._rmax,
                atom_include_keys=("numbers", "forces", *self._tensors),
                # global_include_keys=("energy", "stress"),
            )

            train_graphs = list(map(to_graph, all_train))
            val_graphs = list(map(to_graph, all_val))
            test_graphs = list(map(to_graph, all_test))

            calc_padding = functools.partial(
                gcnn.data.GraphBatcher.calculate_padding,
                batch_size=self._batch_size,
                with_shuffle=True,
            )

            paddings = list(map(calc_padding, (train_graphs, val_graphs, test_graphs)))
            # Calculate the max padding we will need for any of the batches
            self._max_padding = gcnn.data.max_padding(*paddings)

            self.data_train = train_graphs
            self.data_val = val_graphs
            self.data_test = test_graphs

    @override
    def train_dataloader(self) -> reax.DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.data_train is None:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the dataloader"
            )

        return gcnn.data.GraphLoader(
            self.data_train,
            batch_size=self._batch_size,
            padding=self._max_padding,
            pad=True,
        )

    @override
    def val_dataloader(self) -> reax.DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        if self.data_val is None:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the dataloader"
            )

        return gcnn.data.GraphLoader(
            self.data_val,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
        )

    @override
    def test_dataloader(self) -> reax.DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        if self.data_test is None:
            raise reax.exceptions.MisconfigurationException(
                "Must call setup() before requesting the dataloader"
            )

        return gcnn.data.GraphLoader(
            self.data_test,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            padding=self._max_padding,
            pad=True,
        )

    def _extract_tarball(self, tar_name: str) -> pathlib.Path:
        data_dir = pathlib.Path(self._data_dir)
        tar_path = data_dir / tar_name
        output_dir = data_dir / tar_name.split(".")[0]

        # Extract the full tarball
        if not output_dir.exists():
            with tarfile.open(tar_path, "r", encoding="utf-8") as tar:
                tar.extractall(data_dir)  # nosec

        return output_dir


def get_tensors(root_dir: pathlib.Path, tensor_type: str, index: str) -> np.ndarray:
    return np.load(f"{root_dir}/{tensor_type}/{index}.npy")


def get_structures(root_dir: pathlib.Path, tensors: Iterable[str]) -> list[ase.Atoms]:
    structures_dir = pathlib.Path(root_dir) / "structures" / "cif"
    _LOGGER.info("Loading structures from: %s", structures_dir.absolute())

    structures: list[ase.Atoms] = []
    for filename in structures_dir.iterdir():
        try:
            structure = ase.io.read(filename, format="cif")
            structure_number = re.findall(r"\d\d\d\d\d", filename.name)[0]

            for tensor in tensors:
                tens = get_tensors(root_dir, tensor, structure_number)
                structure.arrays[tensor] = tens
        except FileNotFoundError:
            continue

        structures.append(structure)

    return structures


def read_scf(filename) -> ase.Atoms:
    with open(filename, "r", encoding="utf-8") as fileobj:
        _data, card_lines = espresso.read_fortran_namelist(fileobj)

    cell, _ = espresso.get_cell_parameters(card_lines)

    nat = 0
    counting = False
    for line in card_lines:
        if counting:
            if not line.strip():
                break

            nat += 1
        elif line.strip().startswith("ATOMIC_POSITIONS"):
            counting = True

    positions_card = espresso.get_atomic_positions(card_lines, n_atoms=nat, cell=cell)

    symbols = [espresso.label_to_symbol(position[0]) for position in positions_card]
    positions = [position[1] for position in positions_card]

    # TODO: put more info into the atoms object
    # e.g magmom, forces.
    atoms = ase.Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)

    return atoms
