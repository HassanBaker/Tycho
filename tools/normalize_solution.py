import numpy as np
import pandas as pd
from sys import argv
from tools.config import labels


def _normalise_on_question(question_matrix):
    epsilon = 1e-12
    question_matrix = question_matrix / (question_matrix.sum(1, keepdims=True) + epsilon)
    # question_matrix[question_matrix < epsilon] = 0.0
    return question_matrix


def normalize_solution(np_matrix):
    spliced_matrix = [np_matrix[:, 0:3],
                      np_matrix[:, 3:5],
                      np_matrix[:, 5:7],
                      np_matrix[:, 7:9],
                      np_matrix[:, 9:13],
                      np_matrix[:, 13:15],
                      np_matrix[:, 15:18],
                      np_matrix[:, 18:25],
                      np_matrix[:, 25:28],
                      np_matrix[:, 28:31],
                      np_matrix[:, 31:37]
                      ]

    for i in range(len(spliced_matrix)):
        spliced_matrix[i] = _normalise_on_question(spliced_matrix[i])

    # weighting factors
    w1 = 1
    w2 = spliced_matrix[0][:, 1]
    w3 = spliced_matrix[1][:, 1] * w2
    w4 = w3
    w7 = spliced_matrix[0][:, 0]
    w9 = spliced_matrix[1][:, 0] * w2
    w10 = spliced_matrix[3][:, 0] * w4
    w11 = w10
    w5 = w4
    w6 = 1
    w8 = spliced_matrix[5][:, 0] * w6

    w2 = np.reshape(w2, [len(w2), 1])
    w3 = np.reshape(w3, [len(w3), 1])
    w4 = np.reshape(w4, [len(w4), 1])
    w5 = np.reshape(w5, [len(w5), 1])
    w7 = np.reshape(w7, [len(w7), 1])
    w8 = np.reshape(w8, [len(w8), 1])
    w9 = np.reshape(w9, [len(w9), 1])
    w10 = np.reshape(w10, [len(w10), 1])
    w11 = np.reshape(w11, [len(w11), 1])

    # weighted answers
    wq1 = spliced_matrix[0] * w1
    wq2 = spliced_matrix[1] * w2
    wq3 = spliced_matrix[2] * w3
    wq4 = spliced_matrix[3] * w4
    wq5 = spliced_matrix[4] * w5
    wq6 = spliced_matrix[5] * w6
    wq7 = spliced_matrix[6] * w7
    wq8 = spliced_matrix[7] * w8
    wq9 = spliced_matrix[8] * w9
    wq10 = spliced_matrix[9] * w10
    wq11 = spliced_matrix[10] * w11

    output = np.concatenate((
        wq1, wq2, wq3, wq4, wq5, wq6, wq7, wq8, wq9, wq10, wq11),
        axis=1)
    # output[output < 1e-4] = 0.0
    return output


if __name__ == '__main__':
    solutions_file = "/home/hassan/fyp/tycho/gen_files/solutions/tycho_1.2_TRAIN_SELECT_ONE_ADAM_0.0004_relu_5000.csv"
    solutions_df = pd.read_csv(solutions_file)
    solutions_df[labels[1:]] = normalize_solution(solutions_df[labels[1:]].values)
    solutions_df.to_csv(solutions_file[:-4] + "_normalised_no_zeroing.csv", index=False)
    print("Normalization complete")
