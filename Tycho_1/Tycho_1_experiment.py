from tools.model_analyzer import model_analyzer

from Tycho_1_specific_blocks import *


def run_experiments():
    IMAGE_SIZE = 45
    CHANNELS = 3
    NUM_LABELS = 37
    BATCH_SIZE = 16

    ACTIVATION_FUNCTIONS = ["sigmoid", "relu"]
    LEARNING_OPTIMIZERS = ["SGD", "NESTEROV" "ADAM"]
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
