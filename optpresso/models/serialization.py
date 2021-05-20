from keras.models import load_model as keras_load

from optpresso.models.layers import SubtractMeanLayer
from optpresso.models.metrics import (
    psuedo_huber_loss,
    adjusted_mse,
    correlation_coefficient_loss,
)

custom_objects = {
    "SubtractMeanLayer": SubtractMeanLayer,
    "psuedo_huber": psuedo_huber_loss,
    "correlation_loss": correlation_coefficient_loss,
    "adjusted_mse_loss": adjusted_mse,
}


def load_model(path: str, **kwargs):
    return keras_load(path, custom_objects=custom_objects, **kwargs)
