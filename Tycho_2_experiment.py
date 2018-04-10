from tools.model_analyzer import model_analyzer

from tools.network_blocks import *

IMAGE_SIZE = 45
CHANNELS = 3
NUM_LABELS = 37
BATCH_SIZE = 256
NUM_AUGMENTS = 16

NUMBER_OF_CONV_LAYERS = [4, 5]
ACTIVATION_FUNCTIONS_DENSE = ["relu", "maxout"]
ACTIVATION_FUNCTIONS_FINAL = ["relu", "sigmoid"]
LEARNING_OPTIMIZER = "ADAM"
LEARNING_RATE = 0.0004
DROPOUT = [0, 0.5]

TRAINING_EPOCHS = 2000
RECORD_INTERVAL = 100
SAVE_INTERVAL = 100
NAME = "tycho_2.1_experiments"
AUGMENT = "CONCAT"


def create_experimental_network(number_of_conv_layers, activation_function_dense, activation_function_final,
                                dropout_prob=0):
    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE * NUM_AUGMENTS, IMAGE_SIZE, IMAGE_SIZE, CHANNELS],
                       name="input")

    conv = None
    if number_of_conv_layers == 3:
        conv = conv_layer_exp_1(x)
    elif number_of_conv_layers == 4:
        conv = conv_layers(x)
    elif number_of_conv_layers == 5:
        conv = conv_layer_exp_2(x)

    reshaped_conv_ouput = tf.reshape(conv, shape=[-1, 512 * NUM_AUGMENTS])

    fully_connected = None
    if activation_function_dense == "relu":
        fully_connected = tycho_2_fully_connected_layers(reshaped_conv_ouput, dropout_prob=dropout_prob)
    elif activation_function_dense == "maxout":
        fully_connected = maxout_layers(reshaped_conv_ouput, dropout_prob=dropout_prob)

    activation_layer = dense_layer(fully_connected,
                                   weight_shape=[1024, NUM_LABELS],
                                   bias_shape=[NUM_LABELS],
                                   stddev=0.001,
                                   activation=activation_function_final,
                                   name="activation_layer")

    y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS], name="labels")

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
    for activation_function_dense in ACTIVATION_FUNCTIONS_DENSE:

        for activation_function_final in ACTIVATION_FUNCTIONS_FINAL:

            for dropout_prob in DROPOUT:
                for number_of_conv_layers in NUMBER_OF_CONV_LAYERS:

                    summary, train_step, loss, x, y_ = create_experimental_network(number_of_conv_layers,
                                                                                   activation_function_dense,
                                                                                   activation_function_final,
                                                                                   dropout_prob=dropout_prob)

                    _NAME = NAME + "_" + \
                            "CONV" + str(number_of_conv_layers) + "_" + \
                            LEARNING_OPTIMIZER + "_" + \
                            str(LEARNING_RATE) + "_" + \
                            activation_function_dense + "_" + \
                            activation_function_final + "_" + \
                            str(dropout_prob)

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
