import numpy as np
import pytest
import reax

from e3response.data.qm9_nmr import DATASET_URLS, Qm9NmrDataModule, Qm9NmrDataset


@pytest.mark.parametrize("dataset_name", list(DATASET_URLS.keys()))
def test_qm9mrdataset_graphs_contain_expected_keys(dataset_name):
    dataset = Qm9NmrDataset(
        dataset=dataset_name,
        atom_keys=["species", "anisotropy"],
        limit=10,
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
        assert (
            "NMR_tensors" in graph.nodes
        ), f"Graph {i} lacks 'NMR_tensors' for dataset {dataset_name}"
        assert isinstance(
            graph.nodes["mu"], np.ndarray
        ), f"'mu' in graph {i} is not a numpy array for dataset {dataset_name}"
        print(graph.nodes["mu"])


@pytest.mark.parametrize("dataset_name", list(DATASET_URLS.keys()))
def test_qm9_nmr_dataloader_outputs_correct_graphs(dataset_name):
    dm = Qm9NmrDataModule(
        dataset=dataset_name,
        train_val_test_split=(0.6, 0.2, 0.2),
        limit=5,
        batch_size=1,
    )

    class DummyStage(reax.Stage):
        def __init__(self):
            super().__init__(
                name="dummystage",
                module=None,
                strategy=None,
                rng=reax.Generator(seed=42),
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

        # Check mu
        assert "mu" in batch.nodes, f"{loader_fn} batch missing 'mu'"
        mu = batch.nodes["mu"]
        print(mu)

        assert isinstance(mu, np.ndarray), f"'mu' in {loader_fn} is not a numpy array"
        assert mu.ndim == 1, f"'mu' in {loader_fn} has wrong shape {mu.shape}, expected 1D array"
        assert not np.any(np.isnan(mu)), f"'mu' in {loader_fn} contains NaNs"
