import pytest

from optpresso.utils import GroundsLoader


def test_grounds_loader_weighting(tmpdir):
    with tmpdir.as_cwd():
        loader = GroundsLoader(10, (240, 320), directory=".")
        assert len(loader._paths) == 0
        assert loader._weights is None
        loader._paths = [
            (10, "nosuchpath"),
            (10, "nosuchpath"),
            (10, "nosuchpath"),
            (1, "nosuchpath"),
        ]
        calcd_weights = loader.weights
        assert len(calcd_weights) == 11
        for i in range(len(calcd_weights)):
            if i == 1:
                continue
            assert calcd_weights[i] == 1.0
        assert calcd_weights[1] == 5 / 3
