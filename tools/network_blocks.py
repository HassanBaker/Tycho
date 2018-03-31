import pandas as pd
import tensorflow as tf
from tools.config import labels, solutions_dir


def weights(shape, stddev=0.01):
    return tf.Variable(tf.truncated_normal(shape,
                                           stddev=stddev,
                                           name="weight"))


def biases(shape):
    return tf.Variable(tf.constant(0.1,
                                   shape=shape,
                                   name="bias"))


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
        elif activation == "maxout":
            y = tf.nn.sigmoid(tf.matmul(input_tensor, w) + b)
        tf.summary.histogram("weights", w)
        tf.summary.histogram("biases", b)
        tf.summary.histogram("activations", y)
    return y


def maxout_layer(input_tensor, num_units, name="maxout"):
    with tf.name_scope(name):
        layer = tf.contrib.layers.maxout(input_tensor, num_units)
        tf.summary.histogram("maxout", layer)
    return layer


def OLS(y, y_):
    with tf.name_scope("OLS"):
        ols = tf.losses.mean_squared_error(y_, y)
        # ols = tf.reduce_sum((y - y_) * (y - y_))
        tf.summary.scalar("OLS", ols)
    return ols


def Crossentropy(y, y_):
    with tf.name_scope("cross_entropy"):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
        tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy


def SGD(learning_rate, loss):
    with tf.name_scope("train"):
        sgd = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return sgd


def Nesterov(learning_rate, loss, momentum=0.9):
    with tf.name_scope("train"):
        nesterov = tf.train.MomentumOptimizer(learning_rate, momentum, use_nesterov=True).minimize(loss)
    return nesterov


def Adam(learning_rate, loss):
    with tf.name_scope("train"):
        adam = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1=0.9,
                                      beta2=0.999,
                                      epsilon=1e-08,
                                      ).minimize(loss)
    return adam


def RMSE(y, y_):
    with tf.name_scope("error"):
        error = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, y))))
        tf.summary.scalar("RMSE", error)
    return error


def Accuracy(y, y_):
    with tf.name_scope("accuracy"):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    return accuracy


def produce_solutions_csv(predictions_list, NAME, train_iteration):
    solution_path = solutions_dir + NAME + "_" + str(train_iteration) + ".csv"
    prediction_df = pd.DataFrame(predictions_list)
    # prediction_df.drop_duplicates(labels[0])
    prediction_df.to_csv(solution_path, header=labels, index=False)
    print("SOLUTION: ", solution_path)
