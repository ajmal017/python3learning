
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession


def getLogger():

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        filename='/home/gyy/pysparkTry.log',
                        filemode='w',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        )

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    return logging.getLogger(__name__)



def lrmodel(x, label):
    """
        a sigmoid function model
    :param x:
    :param label:
    :return:
    """
    pred = slim.fully_connected(inputs=x, num_outputs=1, activation_fn=tf.nn.sigmoid)
    loss = -tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=pred))
    trainOp = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    return pred, loss, trainOp



def tff(model, getlogger):
    def f(iter):
        logger = getlogger()
        data = []
        for item in iter:
            data.append(item)

        data = np.vstack(data)
        logger.info("data shape is %s %s" % (data.shape[0], data.shape[1]))

        dataX = data[:,:10]
        dataY = data[:,10].reshape([-1, 1])

        x = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        label = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        pred, loss, trainOp = model(x, label)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(10):
                _, l = sess.run([trainOp, loss], feed_dict={x: dataX, label: dataY})
                logger.info("loss is %s" % l)
    return f


def main():
    """
        a simple pyspark + tensorflow test
        it's not the ps, worker mode.

        I will add ps, worker later
    :return:
    """
    # spark = SparkSession.builder.appName("test").master("local").getOrCreate()
    conf = SparkConf().setAppName("test").setMaster("spark://namenode01:7077") \
        .set("spark.shuffle.service.enabled", "false").set("spark.dynamicAllocation.enabled", "false")
        #.set('spark.executor.instances', '1').set('spark.executor.cores', '1')\
        #.set('spark.executor.memory', '512m').set('spark.driver.memory', '512m')\
    # conf = SparkConf().setAppName("test").setMaster("local")

    sc = SparkContext(conf=conf)
    dataX = np.random.random((40,10))
    dataY = np.zeros((40,1))
    dataY[:20,0] = 1
    data = np.split(np.hstack([dataX, dataY]), 40, axis=0)

    rdd = sc.parallelize(data)
    rdd.foreachPartition(tff(lrmodel, getLogger))
    # a = 1

if __name__ == '__main__':
    main()