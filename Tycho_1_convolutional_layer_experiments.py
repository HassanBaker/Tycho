from tools.model_analyzer import model_analyzer

from tools.network_blocks import *

"""
An experiment to analyse the effect of adding an extra convolutional layer and taking away one convolutional layer
"""

IMAGE_SIZE = 45
CHANNELS = 3
NUM_LABELS = 37
BATCH_SIZE = 16

TRAINING_EPOCHS = 2000
RECORD_INTERVAL = 50
SAVE_INTERVAL = 100
NAME = "tycho_1_CONV_EXP"
AUGMENT = "SELECT_ONE"

FINAL_LAYER_ACTIVATION = "relu"
LEARNING_OPTIMIZER = "ADAM"
LEARNING_RATE = 0.0004
NUMBER_OF_CONV_LAYERS = [3, 4, 5]


def create_experimental_network(number_of_conv_layers):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS], name="input")

    if number_of_conv_layers == 3:
        conv = conv_layer_exp_1(x)
        fully_connected = tycho_1_fully_connected_layers(conv)
    elif number_of_conv_layers == 4:
        conv = conv_layers(x)
        fully_connected = tycho_1_fully_connected_layers(conv)
    elif number_of_conv_layers == 5:
        conv = conv_layer_exp_2(x)
        fully_connected = tycho_1_fully_connected_layers(conv)

    activation_layer = dense_layer(fully_connected,
                                   weight_shape=[64, NUM_LABELS],
                                   bias_shape=[NUM_LABELS],
                                   stddev=0.001,
                                   activation="relu",
                                   name="activation_layer")

    y_ = tf.placeholder(tf.float32, shape=[16, NUM_LABELS], name="labels")

    with tf.name_scope("OLS"):
        loss = tf.losses.mean_squared_error(y_, activation_layer)
        tf.summary.scalar("OLS", loss)

    with tf.name_scope("Adam"):
        train_step = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE,
                                            beta1=0.9,
                                            beta2=0.999,
                                            epsilon=1e-08,
                                            ).minimize(loss)

    with tf.name_scope("RMSE"):
        error = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, activation_layer))))
        tf.summary.scalar("RMSE", error)

    summary = tf.summary.merge_all()

    return summary, train_step, loss, x, y_


def run_experiments():
    for number_of_conv_layers in NUMBER_OF_CONV_LAYERS:
        summary, train_step, loss, x, y_ = create_experimental_network(number_of_conv_layers)

        _NAME = NAME + "_" + \
                str(number_of_conv_layers) + "_" + \
                LEARNING_OPTIMIZER + "_" + \
                str(LEARNING_RATE) + "_" + \
                FINAL_LAYER_ACTIVATION

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
