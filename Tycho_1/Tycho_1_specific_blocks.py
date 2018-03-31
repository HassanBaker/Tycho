from tools.config import FULL_TRAIN_DIR, TEST_DIR, log_dir, save_dir
from tools.data_processing import image_data
from tools.network_blocks import *

def conv_layers(input_tensor):
    conv1 = conv_layer(input_tensor,
                       num_channels=3,
                       output_size=32,
                       filter_size=6,
                       pooling=2,
                       name="conv1")

    conv2 = conv_layer(conv1,
                       num_channels=32,
                       output_size=64,
                       filter_size=5,
                       pooling=2,
                       name="conv2")

    conv3 = conv_layer(conv2,
                       num_channels=64,
                       output_size=128,
                       filter_size=3,
                       pooling=None,
                       name="conv3")

    conv4 = conv_layer(conv3,
                       num_channels=128,
                       output_size=128,
                       filter_size=3,
                       pooling=2,
                       name="conv4"
                       )

    flat_layer = tf.contrib.layers.flatten(conv4)

    return flat_layer

def fully_connected_layers(input_tensor):
    dense1 = dense_layer(input_tensor,
                         weight_shape=[512, 128],
                         bias_shape=[128],
                         stddev=0.01,
                         activation="relu",
                         name="dense1")

    dense2 = dense_layer(dense1,
                         weight_shape=[128, 64],
                         bias_shape=[64],
                         stddev=0.01,
                         activation="relu",
                         name="dense2")

    return dense2