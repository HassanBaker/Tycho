from sys import argv

from tqdm import tqdm

from Tycho_1 import conv_layers, fully_connected_layers
from tools.config import TEST_DIR
from tools.data_processing import image_data
from tools.network_blocks import *

IMAGE_SIZE = 45
CHANNELS = 3
NUM_LABELS = 37
BATCH_SIZE = 256
TEST_EPOCHS = 313

ACTIVATION_FUNCTION = "sigmoid"
LEARNING_OPTIMIZER = "ADAM"
LEARNING_RATE = 0.0004

AUGMENT = "SELECT_ONE"

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

sess = tf.Session()
sess.run(tf.global_variables_initializer())
train_iteration = 0

saver = tf.train.Saver()


def _produce_answers_csv(NAME):
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


def _load(load_dir):
    try:
        saver.restore(sess, load_dir)
        try:
            with open(load_dir + "train_iteration", "r") as _file:
                train_it = int(_file.read())
                print("Loaded from train_iteration " + str(train_iteration))
                return train_it
        except Exception:
            print("Cannot read train_iteration")
            return 0
    except Exception:
        print("Cannot read session")
        return 0


def predict(load_dir, output_name):
    _load(load_dir)
    _produce_answers_csv(output_name)


if __name__ == "__main__":
    load_dir = argv[0]
    output_name = argv[1]
    predict(load_dir, output_name)
