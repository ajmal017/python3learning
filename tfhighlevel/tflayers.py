
import tensorflow as tf
import numpy as np
from util.data import getMnist

def cnnTest():

    dataX, dataY = getMnist()

    dataX = dataX.reshape([-1, 20, 20, 1])
    
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.placeholder(dtype=tf.float32, shape=[None, 20, 20, 1])
    label = tf.placeholder(dtype=tf.int32, shape=[None, 10], name="label")

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

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }


    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=label, logits=logits)

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())
    

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(10):
            _, l = sess.run([train_op, loss], feed_dict={input_layer: dataX, label: dataY})

            print(l)



if __name__ == '__main__':
    cnnTest()