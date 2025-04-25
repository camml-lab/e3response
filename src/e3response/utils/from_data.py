import functools
from typing import Mapping, Optional, Union

import hydra
import jaxtyping as jt
import omegaconf
import reax
import reax.utils
from typing_extensions import override

DONE_KEY = "done"


class FromData(reax.stages.Stage):
    def __init__(
        self,
        cfg: omegaconf.DictConfig,
        strategy: reax.Strategy,
        rng: reax.Generator,
        dataloader: Optional[reax.DataLoader] = None,
        datamodule: Optional[reax.DataModule] = None,
        dataloader_name: Optional[str] = "train",
        ignore_missing: bool = True,
    ):
        """
        Populate a hydra configurations dictionary using calculated stats

        :param cfg: the configuration dictionary
        :param strategy: the trainer strategy
        :param rng: the random number generator
        :param dataloader: the dataloader to use
        :param datamodule: if no dataloader is specified, a data module can be used instead
        :param dataloader_name: the datamodule dataloader name
        :param ignore_missing: if `True`, any data that is needed to calculate a metric but is
            missing will be ignored, and that metric will not be calculated
        """
        super().__init__(
            "from_data",
            module=None,
            strategy=strategy,
            rng=rng,
            datamanager=reax.data.create_manager(
                datamodule=datamodule, **{f"{dataloader_name}_loader": dataloader}
            ),
        )
        # Params
        self._dataset_name = dataloader_name
        self._ignore_missing = ignore_missing

        # State
        self._cfg = cfg
        self._dataloader = dataloader
        self._calculated = {}
        self._to_calculate: dict = self._update_stats(self._cfg)

    @property
    def dataloader(self) -> Optional[reax.DataLoader]:
        return self._datamanager.get_dataloader(self._dataset_name)

    @property
    def dataloaders(self) -> Optional[reax.DataLoader]:
        """Dataloader function."""
        return self.dataloader

    @override
    def log(
        self,
        name: str,
        value,
        batch_size: Optional[int] = None,
        prog_bar: bool = False,
        logger: bool = False,
        on_step=False,
        on_epoch=True,
    ) -> None:
        self._child.log(
            name,
            value,
            batch_size=batch_size,
            prog_bar=prog_bar,
            logger=logger,
            on_step=on_step,
            on_epoch=on_epoch,
        )

    @override
    def _step(self) -> None:
        eval_stats = reax.stages.EvaluateStats(
            self._to_calculate,
            self._datamanager,
            self._strategy,
            self._rng,
            dataset_name=self._dataset_name,
            ignore_missing=True,
        )
        calculated: dict = self._run_child(eval_stats).logged_metrics
        # Convert to types that can be used by omegaconf and update the configuration with the values
        calculated = {label: reax.utils.arrays.to_base(stat) for label, stat in calculated.items()}

        if self._ignore_missing:
            # Set any that we couldn't calculate to `None`
            for missing in self._to_calculate.keys() - calculated.keys():
                calculated[missing] = None

        # Update for the next step
        self._cfg.update(calculated)
        self._calculated.update(calculated)

        # Find the next set to get calculated
        self._to_calculate = self._update_stats(self._cfg)

        if not self._to_calculate:
            # we're done
            self._cfg.update(self._calculated)
            self._stopper.set()

    def _update_stats(self, from_data: Mapping) -> dict:
        with_dependencies = []

        # Find those that we will come back to for a second path
        for entry in find_iterpol(from_data):
            with_dependencies.append(entry[0][0])
        with_dependencies = set(with_dependencies)

        to_calculate = {}
        for label, value in from_data.items():
            if label in self._calculated:
                continue

            if label in with_dependencies:
                continue

            if omegaconf.OmegaConf.is_dict(value):
                stat = hydra.utils.instantiate(value, _convert_="object")
            else:
                stat = reax.metrics.get(value)

            to_calculate[label] = stat

        return to_calculate


@functools.singledispatch
def _to_omega(value):
    return value


@_to_omega.register
def _(value: dict) -> dict:
    return {key: _to_omega(value) for key, value in value.items()}


@_to_omega.register
def _(value: jt.Array) -> Union[int, float, list]:
    return reax.utils.arrays.to_base(value)


def find_iterpol(root, path=()):
    for key, value in root.items():
        if isinstance(value, str):
            if omegaconf.OmegaConf.is_interpolation(root, key):
                if not isinstance(root[key], (int, float, list)):
                    yield path, key
        elif omegaconf.OmegaConf.is_dict(value):
            yield from find_iterpol(value, (key,))
