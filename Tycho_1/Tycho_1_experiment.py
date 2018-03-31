from tools.model_analyzer import model_analyzer

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


def run_experiments():
    global train_step
    IMAGE_SIZE = 45
    CHANNELS = 3
    NUM_LABELS = 37
    BATCH_SIZE = 16

    ACTIVATION_FUNCTIONS = ["relu"]
    LEARNING_OPTIMIZERS = ["ADAM"]
    LEARNING_RATES = [0.0004]

    TRAINING_EPOCHS = 2000
    RECORD_INTERVAL = 50
    SAVE_INTERVAL = 100
    NAME = "tycho_1"
    AUGMENT = "SELECT_ONE"

    for activation_function in ACTIVATION_FUNCTIONS:

        for learning_rate in LEARNING_RATES:

            for optimizer in LEARNING_OPTIMIZERS:

                tf.reset_default_graph()

                x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS], name="input")

                conv = conv_layers(x)

                fully_connected = fully_connected_layers(conv)

                activation_layer = dense_layer(fully_connected,
                                               weight_shape=[64, NUM_LABELS],
                                               bias_shape=[NUM_LABELS],
                                               stddev=0.001,
                                               activation=activation_function,
                                               name="activation_layer")

                y_ = tf.placeholder(tf.float32, shape=[16, NUM_LABELS], name="labels")

                with tf.name_scope("OLS"):
                    loss = tf.losses.mean_squared_error(y_, activation_layer)
                    tf.summary.scalar("OLS", loss)

                if optimizer == "SGD":
                    train_step = SGD(learning_rate, loss)
                elif optimizer == "NESTEROV":
                    train_step = Nesterov(learning_rate, loss)
                elif optimizer == "ADAM":
                    with tf.name_scope("Adam"):
                        train_step = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                            beta1=0.9,
                                                            beta2=0.999,
                                                            epsilon=1e-08,
                                                            ).minimize(loss)

                with tf.name_scope("RMSE"):
                    error = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, activation_layer))))
                    tf.summary.scalar("RMSE", error)

                summary = tf.summary.merge_all()

                _NAME = NAME + "_" + optimizer + "_" + str(learning_rate) + "_" + activation_function

                analyzer = model_analyzer(
                    summary,
                    name=_NAME,
                    augment=AUGMENT
                )

                analyzer.train(
                    train_step,
                    loss,
                    x,
                    y_,
                    TRAINING_EPOCHS,
                    BATCH_SIZE,
                    RECORD_INTERVAL,
                    save_interval=SAVE_INTERVAL
                )

                print("\nCompleted - ", _NAME)


if __name__ == "__main__":
    run_experiments()
