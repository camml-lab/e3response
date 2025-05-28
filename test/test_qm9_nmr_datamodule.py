import numpy as np
import pytest
from reax import Generator, Stage

from e3response.data.qm9_nmr import DATASET_URLS, QM9NmrDataModule


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
        assert "mask" in batch.nodes, f"{loader_fn} batch missing 'mask'"

        nmr_tensors = batch.nodes["NMR_tensors"]
        mask = batch.nodes["mask"]

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

        # Check mask
        assert isinstance(mask, np.ndarray), f"'mask' in {loader_fn} is not a numpy array"
        assert mask.dtype == bool, f"'mask' in {loader_fn} is not boolean"
        assert (
            mask.shape[0] == nmr_tensors.shape[0]
        ), f"Mask and NMR_tensors batch/atom dims don't match in {loader_fn}"
