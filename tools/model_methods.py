from tools.network_blocks import *
from tools.config import log_dir, save_dir, solutions_dir, labels
from tqdm import tqdm
import pandas as pd


def produce_solutions_csv(predictions_list, NAME, train_iteration):
    solution_path = solutions_dir + NAME + "_" + str(train_iteration) + ".csv"
    prediction_df = pd.DataFrame(predictions_list)
    # prediction_df.drop_duplicates(labels[0])
    prediction_df.to_csv(solution_path, header=labels, index=False)
    print("SOLUTION: ", solution_path)


def produce_answers_csv(TEST_EPOCHS, TEST_DATA, BATCH_SIZE,
                        session, input_layer, final_layer,
                        NAME, train_iteration):
    print("TESTING")
    TEST_DATA.reset()
    predictions = []
    with tqdm(total=TEST_EPOCHS) as test_pbar:
        for epoch in range(TEST_EPOCHS):
            test_batch_xs, test_batch_ys = TEST_DATA.next_batch(BATCH_SIZE)

            current_predictions = session.run(final_layer,
                                              feed_dict={input_layer: test_batch_xs})
            current_predictions = current_predictions.tolist()
            test_batch_ys = test_batch_ys.tolist()
            for e in range(len(current_predictions)):
                current_predictions[e].insert(0, test_batch_ys[e])
            predictions += current_predictions
            test_pbar.update(1)

    produce_solutions_csv(predictions, NAME, train_iteration)
    print("FINISHED TESTING")


# noinspection PyBroadException
def load(saver, session, LOAD_DIR, train_iteration):
    try:
        saver.restore(session, LOAD_DIR + "_session")
        try:
            with open(LOAD_DIR + "train_iteration", "r") as _file:
                train_it = int(_file.read())
                print("Loaded from train_iteration " + str(train_iteration))
                return train_it
        except Exception:
            print("Cannot read train_iteration")
            return 0
    except Exception:
        print("Cannot read session")
        return 0


def save(saver, session, SAVE_DIR, train_iteration):
    saver.save(session, SAVE_DIR + "_session")
    with open(SAVE_DIR + "train_iteration", "w") as _file:
        _file.write(str(train_iteration))


def get_loss_for_batch(session, loss, summary, writer, train_iteration,
                       batch_xs, batch_ys, input_layer, output_layer):
    _loss, summ = session.run(
        [loss, summary],
        feed_dict={input_layer: batch_xs, output_layer: batch_ys})
    writer.add_summary(summ, train_iteration)
    return _loss


def train(x, y_, final_layer, train_step, learning_rate_ph, LEARNING_RATE, loss, summary,
          NAME, TRAIN, TEST, BATCH_SIZE, TRAINING_DURATION, TEST_EPOCHS,
          RECORD_INTERVAL, TEST_INTERVAL, SAVE_INTERVAL, LOAD=False):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter(log_dir + NAME + "/train", sess.graph)
    train_iteration = 0

    saver = tf.train.Saver()
    SAVE_PATH = save_dir + NAME + "/"

    i = 0

    if LOAD:
        train_iteration = load(saver, sess, SAVE_PATH, train_iteration)
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
                get_loss_for_batch(sess, loss, summary, train_writer, train_iteration,
                                   train_batch_xs, train_batch_ys, x, y_)

            if TEST_INTERVAL != 0 and i != 0 and i % TEST_INTERVAL == 0:
                produce_answers_csv(TEST_EPOCHS, TEST, BATCH_SIZE,
                                    sess, x, final_layer,
                                    NAME, train_iteration)

            train_iteration += 1
            i += 1

            if SAVE_INTERVAL != 0 and i != 0 and i % SAVE_INTERVAL == 0:
                save(saver, sess, SAVE_PATH, train_iteration)

            pbar.update(1)

    print("\nCompleted - ", NAME)


