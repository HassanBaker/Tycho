from tools.network_blocks import *

NUM_AUGMENTS = 16


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


def fully_connected_layers(input_tensor, activation_function="relu"):
    dense1 = dense_layer(input_tensor,
                         weight_shape=[512 * NUM_AUGMENTS, 2048],
                         bias_shape=[2048],
                         stddev=0.01,
                         activation=activation_function,
                         name="dense1")

    dense2 = dense_layer(dense1,
                         weight_shape=[2048, 1024],
                         bias_shape=[1024],
                         stddev=0.01,
                         activation=activation_function,
                         name="dense2")

    return dense2


def maxout_layers(input_tensor):
    maxout_1 = maxout_layer(input_tensor, 2048)
    maxout_2 = maxout_layer(maxout_1, 1024)
    return maxout_2
