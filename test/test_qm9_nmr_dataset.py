import numpy as np
import pytest

from e3response.data.qm9_nmr import DATASET_URLS, QM9NmrDataset


@pytest.mark.parametrize("dataset_name", list(DATASET_URLS.keys()))
def test_QM9NmrDataset_graphs_contain_expected_keys(dataset_name):
    dataset = QM9NmrDataset(
        dataset=dataset_name,
        limit=1,
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
        assert "mask" in graph.nodes, f"Graph {i} lacks 'mask' for dataset {dataset_name}"
        assert isinstance(
            graph.nodes["NMR_tensors"], np.ndarray
        ), f"'NMR_tensors' in graph {i} is not a numpy array for dataset {dataset_name}"
        assert graph.nodes["NMR_tensors"].shape[-2:] == (
            3,
            3,
        ), f"Wrong NMR tensor shape in graph {i} for dataset {dataset_name}"
        assert isinstance(
            graph.nodes["mask"], np.ndarray
        ), f"'mask' in graph {i} is not a numpy array for dataset {dataset_name}"
        assert (
            graph.nodes["mask"].dtype == bool
        ), f"Non boolean 'mask' in graph {i} for dataset {dataset_name}"
