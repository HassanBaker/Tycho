from tools.config import FULL_TRAIN_DIR, TEST_DIR
from tools.data_processing import image_data
from tools.network_blocks import *
from tools.model_methods import train

"""
As Tycho 2 and Tycho 3 share very similar code, the same code is used here.
Hence Tycho 3 is named Tycho2.1, and will appear in the logs, graphs, and session files as Tycho2.1
"""

IMAGE_SIZE = 45
CHANNELS = 3
NUM_LABELS = 37
BATCH_SIZE = 256
NUM_AUGMENTS = 16
TEST_EPOCHS = 313

"""Tycho 2 configuration"""
NAME = "tycho_2_TRAIN"
ACTIVATION_FUNCTION_DENSE = "relu"
ACTIVATION_FUNCTION_FINAL = "relu"

"""Tycho 3 configuration"""
"""
NAME = "tycho_2.1_TRAIN"
ACTIVATION_FUNCTION_DENSE = "maxout"
ACTIVATION_FUNCTION_FINAL = "relu"
"""

LEARNING_OPTIMIZER = "ADAM"
LEARNING_RATE = 0.0004
DROPOUT = 0.5

TRAINING_DURATION = 10000
RECORD_INTERVAL = 100
TEST_INTERVAL = 5000
SAVE_INTERVAL = 100
AUGMENT = "CONCAT"

NAME = NAME + "_" + \
       AUGMENT + "_" + \
       LEARNING_OPTIMIZER + "_" + \
       str(LEARNING_RATE) + "_" + \
       ACTIVATION_FUNCTION_DENSE + "_" + \
       ACTIVATION_FUNCTION_FINAL + "_" + \
       str(DROPOUT)

TRAIN = image_data(FULL_TRAIN_DIR, augment=AUGMENT)
TRAIN.shuffle()

TEST = image_data(TEST_DIR, type="TEST", augment=AUGMENT)
TEST.shuffle()

tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE * NUM_AUGMENTS, IMAGE_SIZE, IMAGE_SIZE, CHANNELS],
                   name="input")

conv = conv_layers(x)

reshaped_conv_ouput = tf.reshape(conv, shape=[-1, 512 * NUM_AUGMENTS])

dense = None
if ACTIVATION_FUNCTION_DENSE == "maxout":
    dense = maxout_layers(reshaped_conv_ouput, dropout_prob=DROPOUT)
elif ACTIVATION_FUNCTION_DENSE == "relu":
    dense = tycho_2_fully_connected_layers(reshaped_conv_ouput, dropout_prob=DROPOUT)

final_layer = dense_layer(dense,
                          weight_shape=[1024, NUM_LABELS],
                          bias_shape=[NUM_LABELS],
                          stddev=0.001,
                          activation=ACTIVATION_FUNCTION_FINAL,
                          name="activation_layer")

y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_LABELS], name="labels")

with tf.name_scope("OLS"):
    loss = tf.losses.mean_squared_error(y_, final_layer)
    tf.summary.scalar("OLS", loss)

learning_rate = tf.placeholder(tf.float32, shape=[])
with tf.name_scope("Adam"):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                        beta1=0.9,
                                        beta2=0.999,
                                        epsilon=1e-08,
                                        ).minimize(loss)

with tf.name_scope("RMSE"):
    error = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(y_, final_layer))))
    tf.summary.scalar("RMSE", error)

summary = tf.summary.merge_all()

if __name__ == "__main__":
    train(x, y_, final_layer, train_step, learning_rate, LEARNING_RATE, loss, summary,
          NAME, TRAIN, TEST, BATCH_SIZE, TRAINING_DURATION, TEST_EPOCHS,
          RECORD_INTERVAL, TEST_INTERVAL, SAVE_INTERVAL, LOAD=True)
