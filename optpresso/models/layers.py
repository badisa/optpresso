from keras.utils import register_keras_serializable
from keras.layers import Layer


@register_keras_serializable()
class SubtractMeanLayer(Layer):
    def __init__(self, mean, **kwargs):
        self.mean = mean
        super(SubtractMeanLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config["mean"] = self.mean
        return config

    def call(self, data):
        return data - self.mean
