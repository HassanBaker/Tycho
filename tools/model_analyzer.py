import tensorflow as tf
from tqdm import tqdm

from tools.config import TRAIN_DIR, VALIDATION_DIR, TEST_DIR, log_dir, save_dir
from tools.data_processing import image_data


class model_analyzer:
    def __init__(self, summary,
                 TRAIN=TRAIN_DIR,
                 VAL=VALIDATION_DIR,
                 TEST=TEST_DIR,
                 name="temp",
                 augment=True,
                 load_on_start=True):
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.train_writer = tf.summary.FileWriter(log_dir + name + "/train", self.sess.graph)
        self.val_writer = tf.summary.FileWriter(log_dir + name + "/val", self.sess.graph)
        self.summary = summary

        self.TRAIN = image_data(TRAIN, augment=augment)
        self.TRAIN.shuffle()

        self.VAL = image_data(VAL, augment=augment)
        self.VAL.shuffle()

        self.TEST = image_data(TEST, augment=augment)
        self.TEST.shuffle()

        self.train_iteration = 0

        self.saver = tf.train.Saver()
        self.SAVE_PATH = save_dir + name + "/"

        if load_on_start:
            self._load()

    def train(self,
              train_step,
              loss,
              x,
              y_,
              epochs,
              batch_size,
              record_interval=0,
              save_interval=0):

        TRAINING_DURATION = epochs
        with tqdm(total=TRAINING_DURATION) as pbar:
            for i in range(TRAINING_DURATION):
                train_batch_xs, train_batch_ys = self.TRAIN.next_batch(batch_size)
                self.sess.run(train_step, feed_dict={x: train_batch_xs, y_: train_batch_ys})

                if record_interval != 0 and i % record_interval == 0:
                    with tf.device('/cpu:0'):
                        val_batch_xs, val_batch_ys = self.VAL.next_batch(batch_size)

                        self._get_loss_for_batch(loss, x, y_, train_batch_xs, train_batch_ys, self.train_writer)

                        self._get_loss_for_batch(loss, x, y_, val_batch_xs, val_batch_ys, self.val_writer)

                self.train_iteration += 1

                if save_interval != 0 and i % save_interval == 0:
                    with tf.device('/cpu:0'):
                        self._save()
                pbar.update(1)

    def _get_loss_for_batch(self, loss, x, y_, batch_xs, batch_ys, writer):
        loss, summ = self.sess.run(
            [loss, self.summary],
            feed_dict={x: batch_xs, y_: batch_ys})
        writer.add_summary(summ, self.train_iteration)
        return loss

    def _save(self):
        self.saver.save(self.sess, self.SAVE_PATH + "_session")
        with open(self.SAVE_PATH + "train_iteration", "w") as _file:
            _file.write(str(self.train_iteration))

    def _load(self):
        try:
            self.saver.restore(self.sess, self.SAVE_PATH + "_session")
            try:
                with open(self.SAVE_PATH + "train_iteration", "r") as _file:
                    self.train_iteration = int(_file.read())
                    print("Loaded from train_iteration " + str(self.train_iteration))
            except Exception:
                print("Cannot read train_iteration")
        except Exception:
            print("Cannot read session")
