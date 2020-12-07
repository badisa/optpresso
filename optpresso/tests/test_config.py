import pytest

import numpy as np

from optpresso.data.config import OptpressoConfig


def test_secondary_model(tmpdir):
    with tmpdir.as_cwd():
        config = OptpressoConfig(
            model="nope", use_secondary_model=False, data_path="nothing.npy"
        )
        with pytest.raises(RuntimeError):
            config.load_secondary_model()
        config.use_secondary_model = True
        config.update_secondary_model([15.0, 15.05], 17.0)
        secondary_model = config.load_secondary_model()
        new_pred = 15.0
        assert secondary_model.predict([[new_pred]]) + new_pred == pytest.approx(17.0)

        # If the value hasn't been seen before, should be 'trusted'
        new_pred = 120.0
        assert secondary_model.predict([[new_pred]]) + new_pred == pytest.approx(
            new_pred
        )
