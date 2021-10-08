"""
The inspiration/code came from the following:

- https://github.com/rsyamil/cnn-regression
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
- https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/rambo
"""
from typing import List
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Activation,
    Flatten,
    Dense,
    Dropout,
    LeakyReLU,
    InputLayer,
    Convolution2D,
    BatchNormalization
)
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers.experimental.preprocessing import (
    Rescaling,
    RandomRotation,
    RandomFlip,
)

from optpresso.models.metrics import (
    correlation_coefficient_loss,
    psuedo_huber_loss,
)
from optpresso.models.layers import SubtractMeanLayer

from optpresso.constants import MEAN_PULL_TIME, MEAN_IMG_VALUES


def create_comma_model_relu(input_shape: List[int]):
    model = Sequential()

    model.add(
        Convolution2D(
            16, (8, 8), strides=(4, 4), padding="same", input_shape=input_shape
        )
    )
    model.add(Activation("relu"))
    model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Activation("relu"))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model_lrelu(input_shape: List[int]):
    model = Sequential()

    model.add(
        Convolution2D(
            16, (8, 8), strides=(4, 4), padding="same", input_shape=input_shape
        )
    )
    model.add(LeakyReLU())
    model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(LeakyReLU())
    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    # model.add(Dropout(.5))
    model.add(LeakyReLU())
    model.add(Dense(512))
    # model.add(Dropout(.5))
    model.add(LeakyReLU())
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model_prelu(input_shape: List[int]):
    model = Sequential()

    model.add(
        Convolution2D(
            16, (8, 8), strides=(4, 4), padding="same", input_shape=input_shape
        )
    )
    model.add(PReLU())
    model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(PReLU())
    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    # model.add(Dropout(.5))
    model.add(PReLU())
    model.add(Dense(512))
    # model.add(Dropout(.5))
    model.add(PReLU())
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model2(input_shape: List[int]):
    # additional dense layer

    model = Sequential()

    model.add(
        Convolution2D(
            16, (8, 8), strides=(4, 4), padding="same", input_shape=input_shape
        )
    )
    model.add(Activation("relu"))
    model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Activation("relu"))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model3(input_shape: List[int]):
    # additional conv layer
    model = Sequential()

    model.add(
        Convolution2D(
            16, (8, 8), strides=(4, 4), padding="same", input_shape=input_shape
        )
    )
    model.add(Activation("relu"))
    model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, 3, 3, padding="same"))
    model.add(Flatten())
    model.add(Activation("relu"))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model4(input_shape: List[int]):
    # 2 additional conv layers
    model = Sequential()

    model.add(
        Convolution2D(
            16, (8, 8), strides=(4, 4), padding="same", input_shape=input_shape
        )
    )
    model.add(Activation("relu"))
    model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, 3, 3, padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, 3, 3, padding="same"))
    model.add(Flatten())
    model.add(Activation("relu"))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model5(input_shape: List[int]):
    # more filters in first 2 conv layers
    model = Sequential()

    model.add(
        Convolution2D(
            32, (8, 8), strides=(4, 4), padding="same", input_shape=input_shape
        )
    )
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Activation("relu"))
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model6(input_shape: List[int]):
    # remove one conv layer
    model = Sequential()

    model.add(
        Convolution2D(
            16, (8, 8), strides=(4, 4), padding="same", input_shape=input_shape
        )
    )
    model.add(Activation("relu"))
    model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Activation("relu"))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model_bn(input_shape: List[int]):
    model = Sequential()

    model.add(
        Convolution2D(
            16, (8, 8), strides=(4, 4), padding="same", input_shape=input_shape
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_nvidia_model1(input_shape: List[int]):
    model = Sequential()

    model.add(
        Convolution2D(
            24, (5, 5), strides=(2, 2), padding="same", input_shape=input_shape
        )
    )
    model.add(Activation("relu"))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Activation("relu"))
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dense(50))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_nvidia_model2(input_shape: List[int]):
    model = Sequential()

    model.add(
        Convolution2D(
            24, (5, 5), strides=(2, 2), padding="same", input_shape=input_shape
        )
    )
    model.add(Activation("relu"))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Activation("relu"))
    model.add(Dense(100))
    model.add(Activation("relu"))
    model.add(Dense(50))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_nvidia_model3(input_shape: List[int]):
    model = Sequential()

    model.add(
        Convolution2D(
            24, (5, 5), strides=(2, 2), padding="same", input_shape=input_shape
        )
    )
    model.add(Activation("relu"))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (3, 3), strides=(2, 2), padding="same"))
    model.add(Flatten())
    model.add(Activation("relu"))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model_large(input_shape: List[int]):
    model = Sequential()

    model.add(
        Convolution2D(
            16, (8, 8), strides=(4, 4), padding="same", input_shape=input_shape
        )
    )
    # model.add(ELU())
    model.add(Activation("relu"))
    model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
    # model.add(ELU())
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    # model.add(Dropout(.5))
    # model.add(ELU())
    model.add(Activation("relu"))
    model.add(Dense(1024))
    # model.add(Dropout(.5))
    # model.add(ELU())
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model_large_dropout(input_shape: List[int]):
    model = Sequential()

    model.add(
        Convolution2D(
            16, (8, 8), strides=(4, 4), padding="same", input_shape=input_shape
        )
    )
    # model.add(ELU())
    model.add(Activation("relu"))
    model.add(Convolution2D(32, (5, 5), strides=(2, 2), padding="same"))
    # model.add(ELU())
    model.add(Activation("relu"))
    model.add(Convolution2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(Flatten())
    # model.add(Dropout(.5))
    # model.add(ELU())
    model.add(Activation("relu"))
    model.add(Dense(1024))
    model.add(Dropout(0.5))
    # model.add(ELU())
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_optpresso_model(input_shape: List[int]) -> Sequential:
    model = Sequential()
    model.add(InputLayer(input_shape=input_shape))
    model.add(SubtractMeanLayer(mean=MEAN_IMG_VALUES))
    model.add(Rescaling(1.0 / 255))
    model.add(RandomFlip())
    model.add(RandomRotation(1))

    model.add(
        Convolution2D(
            32,
            (5, 5),
            padding="same",
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(SpatialDropout2D(0.3))
    model.add(
        Convolution2D(
            48,
            (5, 5),
            strides=(2, 2),
            padding="same",
        )
    )
    # model.add(BatchNormalization())
    # model.add(SpatialDropout2D(0.3))
    model.add(Activation("relu"))
    model.add(
        Convolution2D(
            48,
            (5, 5),
            strides=(2, 2),
            padding="same",
        )
    )
    # model.add(SpatialDropout2D(0.1))
    model.add(Activation("relu"))
    model.add(
        Convolution2D(
            64,
            (3, 3),
            strides=(2, 2),
            padding="same",
        )
    )
    # model.add(SpatialDropout2D(0.1))
    model.add(Activation("relu"))
    model.add(
        Convolution2D(
            64,
            (3, 3),
            strides=(2, 2),
            padding="same",
        )
    )
    # model.add(SpatialDropout2D(0.1))
    model.add(Activation("relu"))
    model.add(
        Convolution2D(
            128,
            (3, 3),
            strides=(2, 2),
            padding="same",
        )
    )
    # model.add(SpatialDropout2D(0.1))
    model.add(Flatten())
    model.add(Activation("relu"))
    model.add(Dense(256))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(64))
    model.add(Dropout(0.5))
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_PULL_TIME)))

    model.compile(
        optimizer=Adam(learning_rate=3e-4),
        loss=psuedo_huber_loss,
        metrics=["mse"],
    )

    return model


MODEL_CONSTRUCTORS = dict(
    optpresso=create_optpresso_model,
    comma_large_dropout=create_comma_model_large_dropout,
    comma_large=create_comma_model_large,
    comma_prelu=create_comma_model_prelu,
    nvidia3=create_nvidia_model3,
    nvidia2=create_nvidia_model2,
    nvidia1=create_nvidia_model1,
    comma_bn=create_comma_model_bn,
    comma6=create_comma_model6,
    comma5=create_comma_model5,
    comma4=create_comma_model4,
    comma3=create_comma_model3,
    comma2=create_comma_model2,
    comma_relu=create_comma_model_relu,
    comma_lrelu=create_comma_model_lrelu,
)
