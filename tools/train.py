from tools.network_blocks import *
from tools.config import log_dir, save_dir
from tqdm import tqdm


def train(x, y_, final_layer, train_step, learning_rate_ph, LEARNING_RATE, loss, summary,
          NAME, TRAIN, TEST, BATCH_SIZE, TRAINING_DURATION, TEST_EPOCHS,
          RECORD_INTERVAL, TEST_INTERVAL, SAVE_INTERVAL, LOAD=False):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(log_dir + NAME + "/train", sess.graph)
    train_iteration = 0

    saver = tf.train.Saver()
    SAVE_PATH = save_dir + NAME + "/"

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

                current_predictions = sess.run(final_layer,
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

    if LOAD:
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
                         learning_rate_ph: LEARNING_RATE})
            if RECORD_INTERVAL != 0 and i % RECORD_INTERVAL == 0:
                _get_loss_for_batch(train_batch_xs, train_batch_ys, train_writer)

            if TEST_INTERVAL != 0 and i != 0 and i % TEST_INTERVAL == 0:
                _produce_answers_csv()

            train_iteration += 1
            i += 1

            if SAVE_INTERVAL != 0 and i != 0 and i % SAVE_INTERVAL == 0:
                _save()

            pbar.update(1)

    print("\nCompleted - ", NAME)
