from tools.network_blocks import *
from tqdm import tqdm

from tools.config import TRAIN_DIR, VALIDATION_DIR, log_dir, save_dir
from tools.data_processing import image_data
from Tycho_1_specific_blocks import *

def run_experiments():
    IMAGE_SIZE = 45
    CHANNELS = 3
    NUM_LABELS = 37
    BATCH_SIZE = 16

    ACTIVATION_FUNCTION = "sigmoid"
    LEARNING_OPTIMIZER = "ADAM"
    LEARNING_RATE = 0.0004
    LEARNING_SCHEDULE = [
        {
            0: 0.0004,
        }
    ]

    TRAINING_EPOCHS = 2000
    RECORD_INTERVAL = 10
    SAVE_INTERVAL = 100
    NAME = "tycho_1_wight"
    AUGMENT = "SELECT_ONE"

    TRAIN = image_data(TRAIN_DIR, augment=AUGMENT)
    TRAIN.shuffle()

    VAL = image_data(VALIDATION_DIR, augment=AUGMENT)
    VAL.shuffle()

    for i in range(len(LEARNING_SCHEDULE)):
        learning_schedule = LEARNING_SCHEDULE[i]

        tf.reset_default_graph()

        x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNELS], name="input")

        conv = conv_layers(x)

        fully_connected = fully_connected_layers(conv)

        activation_layer = dense_layer(fully_connected,
                                       weight_shape=[64, NUM_LABELS],
                                       bias_shape=[NUM_LABELS],
                                       stddev=0.1,
                                       activation=ACTIVATION_FUNCTION,
                                       name="activation_layer")

        y_ = tf.placeholder(tf.float32, shape=[16, NUM_LABELS], name="labels")

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
               str(learning_schedule) + "_" + \
               ACTIVATION_FUNCTION

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        train_writer = tf.summary.FileWriter(log_dir + NAME + "/train", sess.graph)
        val_writer = tf.summary.FileWriter(log_dir + NAME + "/val", sess.graph)
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

        def _get_error_for_batch(batch_xs, batch_ys, writer):
            _error, summ = sess.run(
                [error, summary],
                feed_dict={x: batch_xs, y_: batch_ys})
            writer.add_summary(summ, train_iteration)
            return _error

        def _save():
            saver.save(sess, SAVE_PATH + "_session")
            with open(SAVE_PATH + "train_iteration", "w") as _file:
                _file.write(str(train_iteration))

        with tqdm(total=TRAINING_DURATION) as pbar:
            for i in range(TRAINING_DURATION):
                if i in learning_schedule:
                    LEARNING_RATE = learning_schedule[i]
                    print("\nLearning rate changed to: ", learning_schedule[i])
                train_batch_xs, train_batch_ys = TRAIN.next_batch(BATCH_SIZE)
                sess.run(train_step,
                         feed_dict={
                             x: train_batch_xs,
                             y_: train_batch_ys,
                             learning_rate: LEARNING_RATE})

                if RECORD_INTERVAL != 0 and i % RECORD_INTERVAL == 0:
                    with tf.device('/cpu:0'):
                        val_batch_xs, val_batch_ys = VAL.next_batch(BATCH_SIZE)

                        _get_loss_for_batch(train_batch_xs, train_batch_ys, train_writer)

                        val_error = _get_error_for_batch(val_batch_xs, val_batch_ys, val_writer)
                train_iteration += 1

                if SAVE_INTERVAL != 0 and i % SAVE_INTERVAL == 0:
                    with tf.device('/cpu:0'):
                        _save()
                pbar.update(1)

        print("\nCompleted - ", NAME)


if __name__ == "__main__":
    run_experiments()
