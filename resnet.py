"""CIFAR ResNet-18 and architecture controls."""

from tensorflow import keras
from tensorflow.keras import layers


def build(config):
    penalty = keras.regularizers.l2(config["weight_decay"] / 2)

    def conv(width, kernel, stride=1):
        return layers.Conv2D(width, kernel, strides=stride, padding="same",
                             use_bias=False, kernel_initializer="he_normal",
                             kernel_regularizer=penalty)

    def bn():
        return layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

    inputs = keras.Input((32, 32, 3))
    imagenet = config["stem"] == "imagenet"
    x = layers.ReLU()(bn()(conv(64, 7 if imagenet else 3, 2 if imagenet else 1)(inputs)))
    if imagenet:
        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    for width in (64, 128, 256, 512):
        for block in range(2):
            stride = 2 if block == 0 and width != 64 else 1
            y = layers.ReLU()(bn()(conv(width, 3, stride)(x)))
            y = bn()(conv(width, 3)(y))
            if config["skip"]:
                shortcut = x if stride == 1 else bn()(conv(width, 1, stride)(x))
                y = layers.Add()([y, shortcut])
            x = layers.ReLU()(y)
    x = layers.GlobalAveragePooling2D()(x)
    logits = layers.Dense(10, dtype="float32", kernel_regularizer=penalty)(x)
    return keras.Model(inputs, logits, name="resnet18_cifar")
