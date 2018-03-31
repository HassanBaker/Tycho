from tqdm import tqdm

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


def train(load=False):
    IMAGE_SIZE = 45
    CHANNELS = 3
    NUM_LABELS = 37
    BATCH_SIZE = 16
    TEST_EPOCHS = 313

    ACTIVATION_FUNCTION = "sigmoid"
    LEARNING_OPTIMIZER = "ADAM"
    LEARNING_RATE = 0.0004

    TRAINING_EPOCHS = 10000
    RECORD_INTERVAL = 100
    TEST_INTERVAL = 1000
    SAVE_INTERVAL = 100
    NAME = "tycho_1_TRAIN"
    AUGMENT = "SELECT_ONE"

    TRAIN = image_data(FULL_TRAIN_DIR, augment=AUGMENT)
    TRAIN.shuffle()

    TEST = image_data(TEST_DIR, type="TEST", augment=AUGMENT)
    TEST.shuffle()

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS], name="input")

    conv = conv_layers(x)

    fully_connected = fully_connected_layers(conv)

    activation_layer = dense_layer(fully_connected,
                                   weight_shape=[64, NUM_LABELS],
                                   bias_shape=[NUM_LABELS],
                                   stddev=0.001,
                                   activation=ACTIVATION_FUNCTION,
                                   name="activation_layer")

    activation_layer = tf_output_normalisation_layer(activation_layer)

    y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS], name="labels")

    with tf.name_scope("OLS"):
        loss = tf.losses.mean_squared_error(y_, activation_layer)
        tf.summary.scalar("OLS", loss)

    learning_rate = tf.placeholder(tf.float32, shape=[])
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

    NAME = NAME + "_" + \
           AUGMENT + "_" + \
           LEARNING_OPTIMIZER + "_" + \
           str(LEARNING_RATE) + "_" + \
           ACTIVATION_FUNCTION

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(log_dir + NAME + "/train", sess.graph)
    train_iteration = 0

    saver = tf.train.Saver()
    SAVE_PATH = save_dir + NAME + "/"

    TRAINING_DURATION = TRAINING_EPOCHS

    def _get_loss_for_batch(batch_xs, batch_ys, writer):
        _loss, summ = sess.run(
            [loss, summary],
            feed_dict={x: batch_xs, y_: batch_ys})
        writer.add_summary(summ, train_iteration)
        return _loss

    def _save():
        saver.save(sess, SAVE_PATH + "_session")
        with open(SAVE_PATH + "train_iteration", "w") as _file:
            _file.write(str(train_iteration))

    def _produce_answers_csv():
        print("TESTING")
        predictions = []
        with tqdm(total=TEST_EPOCHS) as test_pbar:
            for epoch in range(TEST_EPOCHS):
                test_batch_xs, test_batch_ys = TEST.next_batch(BATCH_SIZE)

                current_predictions = sess.run(activation_layer,
                                               feed_dict={x: test_batch_xs})
                current_predictions = current_predictions.tolist()
                test_batch_ys = test_batch_ys.tolist()
                for e in range(len(current_predictions)):
                    current_predictions[e].insert(0, test_batch_ys[e])
                predictions += current_predictions
                test_pbar.update(1)

        produce_solutions_csv(predictions, NAME, train_iteration)
        print("FINISHED TESTING")

    def _load():
        try:
            saver.restore(sess, SAVE_PATH + "_session")
            try:
                with open(SAVE_PATH + "train_iteration", "r") as _file:
                    train_it = int(_file.read())
                    print("Loaded from train_iteration " + str(train_iteration))
                    return train_it
            except Exception:
                print("Cannot read train_iteration")
                return 0
        except Exception:
            print("Cannot read session")
            return 0

    i = 0

    if load:
        train_iteration = _load()
        print("train iteration is:", train_iteration)
        print("i is:", i)
        i = train_iteration
        print("changed i to ", i)
        TRAINING_DURATION += train_iteration

    with tqdm(total=TRAINING_DURATION) as pbar:

        pbar.update(train_iteration)

        while i < TRAINING_DURATION:
            train_batch_xs, train_batch_ys = TRAIN.next_batch(BATCH_SIZE)
            sess.run(train_step,
                     feed_dict={
                         x: train_batch_xs,
                         y_: train_batch_ys,
                         learning_rate: LEARNING_RATE})

            if RECORD_INTERVAL != 0 and i != 0 and i % RECORD_INTERVAL == 0:
                _get_loss_for_batch(train_batch_xs, train_batch_ys, train_writer)

            if TEST_INTERVAL != 0 and i != 0 and i % TEST_INTERVAL == 0:
                _produce_answers_csv()

            train_iteration += 1
            i += 1

            if SAVE_INTERVAL != 0 and i != 0 and i % SAVE_INTERVAL == 0:
                _save()

            pbar.update(1)

    print("\nCompleted - ", NAME)


if __name__ == "__main__":
    train(load=True)
