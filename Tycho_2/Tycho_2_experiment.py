from tools.model_analyzer import model_analyzer

from tools.network_blocks import *

def run_experiments():
    global train_step
    IMAGE_SIZE = 45
    CHANNELS = 3
    NUM_LABELS = 37
    BATCH_SIZE = 16
    NUM_AUGMENTS = 16

    ACTIVATION_FUNCTIONS_DENSE = ["relu", "maxout"]
    ACTIVATION_FUNCTIONS_FINAL = ["relu", "sigmoid"]
    LEARNING_OPTIMIZER = "ADAM"
    LEARNING_RATE = 0.0004

    TRAINING_EPOCHS = 2000
    RECORD_INTERVAL = 10
    SAVE_INTERVAL = 100
    NAME = "tycho_1_experiments"
    AUGMENT = "CONCAT"

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

    for activation_function_dense in ACTIVATION_FUNCTIONS_DENSE:

        for activation_function_final in ACTIVATION_FUNCTIONS_FINAL:
            tf.reset_default_graph()

            x = tf.placeholder(tf.float32, shape=[BATCH_SIZE * NUM_AUGMENTS, IMAGE_SIZE, IMAGE_SIZE, CHANNELS],
                               name="input")

            conv = conv_layers(x)

            conv = tf.reshape(conv, shape=[-1, 512 * NUM_AUGMENTS])

            if activation_function_dense == "relu":
                fully_connected = fully_connected_layers(conv, activation_function=activation_function_dense)
            elif activation_function_dense == "maxout":
                fully_connected = maxout_layers(conv)

            activation_layer = dense_layer(fully_connected,
                                           weight_shape=[1024, NUM_LABELS],
                                           bias_shape=[NUM_LABELS],
                                           stddev=0.001,
                                           activation=activation_function_final,
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

            _NAME = NAME + "_" + \
                    LEARNING_OPTIMIZER + "_" + \
                    str(LEARNING_RATE) + "_" + \
                    activation_function_dense + "_" + \
                    activation_function_final

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
