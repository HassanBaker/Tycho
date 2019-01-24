import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt


"""
    MNIST_DATA:
        Shape:
            [ [55000, 784], [55000, 10] ] -> [ 55000, 784, 10 ];
                -> Vector of images' vectors and labels' vectors.

        Images:
            Shape:
                [5500, 784];
                    -> Vector of image index and a scalar of the image data
                    -> 784 is the flattened [28, 28] shape vector for images

        Labels:
            Shape:
                [55000, 10];
                    -> Vector of image index and a scalar of length 10 for the value of the digit in the image;
                    -> Uses a one-hot vector, meaning that the label values are an array nine 0's and one 1.
                        -> The 1 mapped to the index is the value of the digit in the image

        Example:
            Description:
                Entry in MNIST_DATA e.g. MNIST_DATA[i]

            Shape:
                [784, 10];
"""
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
    Input:
        Shape:
            [None, 784];
                -> None implies *any length*;
                    -> Training will be done in batches.
                -> 784 relates to the number of dimensions a flattened image vector has.
                    -> e.g. size of image array
"""
x = tf.placeholder(tf.float32, [None, 784])

"""
    Weight:
        (Learned; initially random)
        Applied to -> Input
        Shape:
            [784, 10];
                -> Shape of example in MNIST_DATA
"""
W = tf.Variable(tf.zeros([784, 10]))

"""
    Bias:
        (Learned; initially random)
        Applied to -> output of Input * Weight;
        Shape:
            [10];
                -> Shape of an example in [Weight * Input];
"""
b = tf.Variable(tf.zeros([10]))

"""
    Output:
        (Equation)
        Shape:
            [100, 10];
                ->

        Functions:
            exp         := [ [ e^_input[j][i] for i in _input[j] ] for j in _input ]
            reduce_sum  := [ sum(i) for i in _input[DIM=-1] ]
            softmax     := exp(_input) / reduce_sum(exp(_input))
"""
y = tf.nn.softmax(tf.matmul(x, W) + b)

"""
    Correct Answer:
        (Given)
        Shape:
            [None, 10];
                -> None implies *any length*;
                    -> Training will be done in batches.
                -> 10 refers to the value of a label as a hot-one vector.
"""
y_ = tf.placeholder(tf.float32, [None, 10])

"""
    Loss:
        Shape:

        Functions:
            reduce_sum      := [ sum(i) for i in _input[DIM = 1] ]
            reduce_mean     := [ avg(i) for i in _input[DIM=-1] ]
            log             := log_e(_input)
            cross_entropy   := reduce_mean( - reduce_sum ( correct * log(incorrect) )

"""
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()
# print(W.eval(sess), W.shape)

def printf(name, input, eval=False, alt=None):
    print(name + ":", input, "\n", "Shape: ", input.shape)
    if eval:
        print("\n", input.eval(), "\n\n")
    else:
        print("\n", alt)
    print("\n\n")

acc = []
__x = []

def plot():

    _x = np.array(__x)
    # axis_l.reshape(len(axis_l))
    plt.plot(acc, _x)
    plt.axis([min(acc), max(acc), 1, max(_x)])
    plt.show()

def test1():
    batch_xs, batch_ys = mnist.train.next_batch(100)

    output = sess.run(y, feed_dict={x: batch_xs})
    printf("Output", output)
    printf("Input", x, alt=batch_xs)
    printf("Weight", W, True)
    printf("Bias", b, True)


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  acc.append(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
  __x.append(i)

plot()






