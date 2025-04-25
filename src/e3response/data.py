import collections
import functools
import numbers
import os
import random
from typing import Callable, Sequence, Union

import ase.io
import jraph
import omegaconf
import reax
from tensorial import gcnn
import tensorial.data
from typing_extensions import override

Datasets = dict[str, gcnn.data.GraphLoader]


def load_datasets_ase(
    paths: dict[str, os.PathLike],
    read_function=ase.io.read,
    limits: dict[str, int] = None,
    **kwargs,
) -> dict[str, list[jraph.GraphsTuple]]:
    to_graph = functools.partial(gcnn.atomic.graph_from_ase, **kwargs)

    datasets = {}
    for name, path in paths.items():
        ase_structures = read_function(path, index=":")
        if limits and name in limits:
            ase_structures = ase_structures[: limits[name]]

        datasets[name] = list(map(to_graph, ase_structures))

    return datasets


def load_datasets_ase_split(
    path: Union[os.PathLike, list[os.PathLike]],
    splits: dict[str, numbers.Number],
    read_function=ase.io.read,
    randomise=True,
    **kwargs,
) -> dict[str, list[jraph.GraphsTuple]]:
    datasets = collections.defaultdict(list)

    if not isinstance(path, list):
        path = [path]

    for entry in path:
        to_graph: Callable[[ase.Atoms], jraph.GraphsTuple] = functools.partial(
            gcnn.atomic.graph_from_ase,
            **kwargs,
        )
        ase_structures = read_function(entry, index=":")
        if randomise:
            random.shuffle(ase_structures)

        graphs = list(map(to_graph, ase_structures))
        total = len(graphs)
        split_total = sum(splits.values())

        num_taken = 0
        for name, split in splits.items():
            num = int(split / split_total * total)
            datasets[name].extend(graphs[num_taken : num_taken + num])
            num_taken += num

    return datasets


def create_batches(
    name: str, graphs: Sequence, batch_size: int, shuffle: bool
) -> gcnn.data.GraphLoader:
    if name == "training":
        # Wrap the training in a caching loader so that we only shuffle every so often
        # (not every epoch)
        dataset = tensorial.data.CachingLoader(
            gcnn.data.GraphLoader(graphs, None, batch_size=batch_size, shuffle=shuffle, pad=True),
            reset_every=10,
        )
    else:
        # For the validation set we cache the whole thing as it never gets shuffled
        dataset = tuple(
            gcnn.data.GraphLoader(graphs, None, batch_size=batch_size, shuffle=False, pad=True)
        )

    return dataset


def create_data_module(
    cfg: omegaconf.DictConfig, datasets: dict[str, tuple[jraph.GraphsTuple]]
) -> reax.DataModule:
    return DataModule(
        datasets,
        batch_size=cfg.batch_size,
        shuffle=cfg.shuffle,
        shuffle_every=cfg.get("shuffle_every", 1),
    )


class DataModule(reax.DataModule):
    def __init__(
        self,
        datasets: dict[str, Sequence[jraph.GraphsTuple]],
        batch_size: int,
        shuffle: bool,
        shuffle_every=1,
    ):
        self._datasets = datasets
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._shuffle_every = shuffle_every

        # Precalculate a padding that will work for all the datasets.  Only shuffle the training
        paddings = [
            gcnn.data.GraphBatcher.calculate_padding(
                graphs, self._batch_size, with_shuffle=name == "training"
            )
            for name, graphs in datasets.items()
        ]
        self._max_padding = gcnn.data.max_padding(*paddings)

    @override
    def train_dataloader(self) -> gcnn.data.GraphLoader:
        graphs = self._datasets["training"]
        loader = gcnn.data.GraphLoader(
            graphs,
            None,
            batch_size=self._batch_size,
            shuffle=self._shuffle,
            pad=True,
            padding=self._max_padding,
        )
        if self._shuffle_every != 1:
            # Wrap the training in a caching loader so that we only shuffle every so often
            # (not every epoch)
            loader = tensorial.data.CachingLoader(loader, reset_every=self._shuffle_every)

        return loader

    @override
    def val_dataloader(self) -> gcnn.data.GraphLoader:
        graphs = self._datasets["validation"]
        loader = gcnn.data.GraphLoader(
            graphs,
            None,
            batch_size=self._batch_size,
            shuffle=False,
            pad=True,
            padding=self._max_padding,
        )
        return loader

    @override
    def predict_dataloader(self) -> gcnn.data.GraphLoader:
        graphs = self._datasets["predict"]
        loader = gcnn.data.GraphLoader(
            graphs,
            None,
            batch_size=self._batch_size,
            shuffle=False,
            pad=True,
            padding=self._max_padding,
        )
        return loader
