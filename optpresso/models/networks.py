"""
The inspiration/code came from the following:

- https://github.com/rsyamil/cnn-regression
- https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
- https://github.com/udacity/self-driving-car/tree/master/steering-models/community-models/rambo
"""
from typing import List, Any
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import (
    Activation,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
    LeakyReLU,
    Lambda,
    ELU,
    SpatialDropout2D,
)
from keras.initializers import Constant, glorot_normal

from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

from optpresso.models.metrics import adjusted_mse

MEAN_VALUE = 30


def create_comma_model_relu(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model_lrelu(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model_prelu(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model2(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model3(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model4(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model5(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model6(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model_bn(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_nvidia_model1(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_nvidia_model2(input_shape: List[int], alpha: float = 0.3):
    model = Sequential()

    model.add(
        Convolution2D(
            24,
            (5, 5),
            padding="same",
            input_shape=input_shape,
            kernel_initializer=glorot_normal(),
        )
    )
    model.add(BatchNormalization())
    model.add(Activation("relu"))
    # model.add(SpatialDropout2D(0.3))
    model.add(
        Convolution2D(
            36,
            (5, 5),
            strides=(2, 2),
            padding="same",
            kernel_initializer=glorot_normal(),
        )
    )
    model.add(BatchNormalization())
    model.add(SpatialDropout2D(0.3))
    model.add(Activation("relu"))
    model.add(
        Convolution2D(
            48,
            (5, 5),
            strides=(2, 2),
            padding="same",
            kernel_initializer=glorot_normal(),
        )
    )
    model.add(SpatialDropout2D(0.3))
    model.add(Activation("relu"))
    model.add(
        Convolution2D(
            64,
            (3, 3),
            strides=(2, 2),
            padding="same",
            kernel_initializer=glorot_normal(),
        )
    )
    model.add(SpatialDropout2D(0.4))
    model.add(Activation("relu"))
    model.add(
        Convolution2D(
            64,
            (3, 3),
            strides=(2, 2),
            padding="same",
            kernel_initializer=glorot_normal(),
        )
    )
    model.add(SpatialDropout2D(0.5))
    model.add(Activation("relu"))
    model.add(
        Convolution2D(
            64,
            (3, 3),
            strides=(2, 2),
            padding="same",
            kernel_initializer=glorot_normal(),
        )
    )
    model.add(SpatialDropout2D(0.5))
    model.add(Flatten())
    model.add(Activation("relu"))
    model.add(Dense(120))
    model.add(Dropout(0.2))
    model.add(Activation("relu"))
    model.add(Dense(50))
    model.add(Activation("relu"))
    model.add(Dense(10))
    model.add(Activation("relu"))
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(
        optimizer=Adam(learning_rate=2e-4), loss="mse", metrics=[adjusted_mse]
    )

    return model


def create_nvidia_model3(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model_large(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_comma_model_large_dropout(input_shape: List[int], alpha: float = 0.3):
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
    model.add(Dense(1, bias_initializer=Constant(MEAN_VALUE)))

    model.compile(optimizer=Adam(learning_rate=3e-4), loss="mse")

    return model


def create_optpresso_model(shape: List[int], alpha: float = 0.3) -> List[Any]:
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), input_shape=shape))
    model.add(LeakyReLU(alpha=alpha))

    model.add(Conv2D(64, (4, 4), strides=(2, 2)))
    model.add(LeakyReLU(alpha=alpha))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (2, 2), strides=(1, 1)))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Flatten())
    # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(128))
    model.add(LeakyReLU(alpha=alpha))

    model.add(Dense(128))
    model.add(LeakyReLU(alpha=alpha))
    model.add(Dropout(0.5))

    # Dense layer of size 1 with linear activation to get that glorious regression
    model.add(
        Dense(1, bias_initializer=Constant(MEAN_VALUE))
    )  # Initialize last layer with aprox mean of data
    model.add(Activation("linear"))
    # A low learning rate seems better, at least when data was ~100 images
    opt = Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss="mse", metrics=["MeanAbsoluteError"])
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
