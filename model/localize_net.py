import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import BatchNormalization, Activation, Conv2D
from tensorflow.keras.models import Sequential

import yaml

def get_config(config_path):
    with open(config_path) as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    return train_config

TRAIN_CONFIG_PATH = 'config.yaml'
train_config = get_config(TRAIN_CONFIG_PATH)

# get config params
IMAGE_HEIGHT = train_config['IMAGE_HEIGHT']
IMAGE_WIDTH = train_config['IMAGE_WIDTH']
WEIGHT_DECAY = train_config['WEIGHT_DECAY']

class LocalizeNet():

    def __init__(self):

        self.backbone = MobileNetV2(input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3), include_top=False, weights="imagenet")

        self.model = Sequential([
                                    Conv2D(112, padding="same", kernel_size=3, strides=1),
                                    BatchNormalization(),
                                    Activation('relu'),
                                    Conv2D(112, padding="same", kernel_size=3, strides=1),
                                    BatchNormalization(),
                                    Activation('relu'),
                                    Conv2D(56, padding="same", kernel_size=3, strides=1, use_bias=False),
                                    BatchNormalization(),
                                    Activation('relu'),
                                    Conv2D(5, padding="same", kernel_size=1, activation="sigmoid")
        ], name='localize_net')

        self.localize_net = None

        pass

    def build(self, trainable):

        for layer in self.backbone.layers:
            layer.trainable = trainable

        feature_map = self.backbone.get_layer("block_16_project_BN").output
        output = self.model(feature_map)

        localize_net = Model(inputs=self.backbone.input, outputs=output)

        # divide by 2 since d/dweight learning_rate * weight^2 = 2 * learning_rate * weight
        # see https://arxiv.org/pdf/1711.05101.pdf
        regularizer = l2(WEIGHT_DECAY / 2)
        for weight in localize_net.trainable_weights:
            with tf.keras.backend.name_scope("weight_regularizer"):
                localize_net.add_loss(lambda: regularizer(weight))  # in tf2.0: lambda: regularizer(weight)

        self.localize_net = localize_net

        return localize_net

if __name__ == '__main__':
    model = LocalizeNet().build(trainable=False)
    model.summary()


