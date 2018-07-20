

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from util.data import getMnist

def tfeTest():
    tfe.enable_eager_execution()

    def square(x):
        return tf.multiply(x, x)

    grad = tfe.gradients_function(square)

    print(square(3.))  # [9.]
    print(grad(3.))

def tfeTestTrain():
    """
        Test how to use tfe with tf.layers
    :return:
    """
    dataX, dataY = getMnist()

    dataX = dataX.reshape([-1,20,20,1])
    input = tf.Variable(initial_value=dataX, dtype=tf.float32)
    cnn = cnnModel(input)
    print(cnn)


def cnnModel(input_layer):
    """Model function for CNN."""
    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    shape = pool2.shape.as_list()
    flatten_size = np.prod(np.array(shape[1:]))
    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, int(flatten_size)])
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=True)

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)

    return logits

if __name__ == '__main__':
    tfeTestTrain()