import tensorflow as tf

"""
Creates a weights variable
"""


def weights(shape, stddev=0.01):
    return tf.Variable(tf.truncated_normal(shape,
                                           stddev=stddev,
                                           name="weight"))


"""
Creates biases variables
"""


def biases(shape):
    return tf.Variable(tf.constant(0.1,
                                   shape=shape,
                                   name="bias"))


"""
Creates a single convolutional layer
If followed by a pooling layer of size X*X, pooling=X, else pooling=None
"""


def conv_layer(input_tensor, num_channels, output_size,
               filter_size, pooling=None, name="conv"):
    with tf.name_scope(name):
        w = weights(shape=[
            filter_size,
            filter_size,
            num_channels,
            output_size
        ])

        b = biases([output_size])

        conv = tf.nn.conv2d(input=input_tensor,
                            filter=w,
                            strides=[1, 1, 1, 1],
                            padding='VALID')

        act = tf.nn.relu(conv + b)

        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", act)
        if pooling is not None:
            conv = tf.layers.max_pooling2d(conv, pooling, pooling)

    return conv


"""
Creates a single dense layer.
Supports relu, sigmoid, and softmax activation.
"""


def dense_layer(input_tensor, weight_shape, bias_shape, stddev=0.01, activation="relu", name="dense"):
    with tf.name_scope(name):
        w = weights(weight_shape, stddev=stddev)
        b = biases(bias_shape)
        if activation == "relu":
            y = tf.nn.relu(tf.matmul(input_tensor, w) + b)
        elif activation == "softmax":
            y = tf.nn.softmax(tf.matmul(input_tensor, w) + b)
        elif activation == "sigmoid":
            y = tf.nn.sigmoid(tf.matmul(input_tensor, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", y)
    return y


"""
Creates a single maxout layer.
"""


def maxout_layer(input_tensor, num_units, name="maxout"):
    with tf.name_scope(name):
        layer = tf.contrib.layers.maxout(input_tensor, num_units)
        tf.summary.histogram("maxout", layer)
    return layer


"""
Creates 4 convolutional layers that are used in Tycho 1 & 2
"""


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


"""
Creates the fully connected layers (except the last layer) for Tycho 1
"""


def tycho_1_fully_connected_layers(input_tensor):
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


"""
Creates the fully connected layers (except the last layer) for Tycho 2.2
"""


def tycho_2_fully_connected_layers(input_tensor, activation_function="relu"):
    dense1 = dense_layer(input_tensor,
                         weight_shape=[512 * 16, 2048],
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


"""
Creates the maxout layers for Tycho 2.1
As described by Sander Dieleman in http://benanne.github.io/2014/04/05/galaxy-zoo.html
"""


def maxout_layers(input_tensor):
    maxout_1 = maxout_layer(input_tensor, 2048)
    maxout_2 = maxout_layer(maxout_1, 1024)
    return maxout_2
