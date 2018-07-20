

import tensorflow as tf
import numpy as np

from tensorflow.contrib.rnn import LSTMCell
from util.data import getMnist

INIT_VAL = 0.001

def mkFakeSeq(data, steps, t):
    fake_seq_X = []
    for i in range(steps):
        fake_seq_X.append(data[i*t:(i+1)*t])
    return np.array(fake_seq_X)

def dynamicrnn():
    """

    :return:
    """
    dataX, dataY = getMnist()

    index = np.arange(dataX.shape[0])
    np.random.shuffle(index)
    dataX = dataX[index]
    dataY = dataY[index]

    t1 = 50
    t2 = 40
    fake_seq_X = []
    fake_seq_Y = []
    steps1 = dataX.shape[0] // t1
    steps2 = dataX.shape[0] // t2

    fake_seq_X1 = mkFakeSeq(dataX, steps1, t1)
    fake_seq_X2 = mkFakeSeq(dataX, steps2, t2)
    # fake_seq_Y = np.array(fake_seq_Y)

    outputSize = 10
    numEmbedding = 128
    seqX = tf.placeholder(dtype=tf.float32, shape=[None, None, 400])
    softmax_w = tf.Variable(tf.random_uniform([outputSize, numEmbedding],
                                                   minval=-INIT_VAL,
                                                   maxval=INIT_VAL))
    softmax_b = tf.Variable(tf.constant(0.0, shape=[numEmbedding]))

    dysteps = tf.placeholder(dtype=tf.int32, shape=[1])
    cell = LSTMCell(numEmbedding)
    outputs, states = tf.nn.dynamic_rnn(cell, seqX, dtype=tf.float32)

    outputsList = tf.split(outputs, dysteps, 1)
    logits = tf.stack([tf.matmul(tf.reshape(t, [-1, outputSize]),
                                                    softmax_w) + softmax_b for t in outputsList])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        l1 = sess.run(logits, feed_dict={seqX:fake_seq_X1, dysteps: np.array([t1])})
        l2 = sess.run(logits, feed_dict={seqX:fake_seq_X2, dysteps: np.array([t2])})
        print(l1)
        print(l2)

if __name__ == '__main__':
    dynamicrnn()