import numpy as np
import pytest
from reax import Generator, Stage

from e3response.data.qm9_nmr import DATASET_URLS, QM9NmrDataModule, QM9NmrDataset


@pytest.mark.parametrize("dataset_name", list(DATASET_URLS.keys()))
def test_qm9mrdataset_graphs_contain_expected_keys(dataset_name):
    dataset = QM9NmrDataset(
        dataset=dataset_name,
        atom_keys=["species", "anisotropy"],
        limit=5,
    )
    assert len(dataset) > 0

    for i, graph in enumerate(dataset):
        assert graph is not None, f"Graph {i} is None for dataset {dataset_name}"
        assert hasattr(
            graph, "nodes"
        ), f"Graph {i} contains no attribute 'nodes' for dataset {dataset_name}"
        assert (
            "NMR_tensors" in graph.nodes
        ), f"Graph {i} lacks 'NMR_tensors' for dataset {dataset_name}"
        assert isinstance(
            graph.nodes["NMR_tensors"], np.ndarray
        ), f"'NMR_tensors' in graph {i} is not a numpy array for dataset {dataset_name}"
        assert graph.nodes["NMR_tensors"].shape[-2:] == (
            3,
            3,
        ), f"Wrong NMR tensor shape in graph {i} for dataset {dataset_name}"


@pytest.mark.parametrize("dataset_name", list(DATASET_URLS.keys()))
def test_qm9_nmr_dataloader_outputs_correct_graphs(dataset_name):
    dm = QM9NmrDataModule(
        dataset=dataset_name,
        limit=20,
        batch_size=8,
    )

    class DummyStage(Stage):
        def __init__(self):
            super().__init__(
                name="dummystage",
                module=None,
                strategy=None,
                rng=Generator(seed=42),
            )

        def _step(self, batch, state):
            return {}

        def log(self, state, step_outputs):
            pass

    dm.setup(DummyStage())

    for loader_fn in ["train_dataloader", "val_dataloader", "test_dataloader"]:
        loader = getattr(dm, loader_fn)()
        batch_tuple = next(iter(loader))

        assert isinstance(batch_tuple, tuple), f"{loader_fn} output is not a tuple"
        batch = batch_tuple[0]

        assert hasattr(batch, "nodes"), f"{loader_fn} batch has no 'nodes'"

        assert "NMR_tensors" in batch.nodes, f"{loader_fn} batch missing 'NMR_tensors'"

        nmr_tensors = batch.nodes["NMR_tensors"]

        # Shape
        assert isinstance(
            nmr_tensors, np.ndarray
        ), f"'NMR_tensors' in {loader_fn} is not a numpy array"
        assert (
            nmr_tensors.ndim == 3
        ), f"'NMR_tensors' in {loader_fn} has wrong shape {nmr_tensors.shape}"
        assert nmr_tensors.shape[-2:] == (
            3,
            3,
        ), f"Last dims of 'NMR_tensors' must be (3,3), got {nmr_tensors.shape[-2:]}"
