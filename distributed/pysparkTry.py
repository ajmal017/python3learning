
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import pyspark
import requests
import time
import random
import multiprocessing
import datetime
import math
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from util.data import getMnist


def getLogger():

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        # filename='/home/gyy/pysparkTry.log',
                        # filemode='w',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        )
    return logging.getLogger(__name__)


def getClusterAndServer(jobName, taskIndex):
    """
        The spec is fix for this demo,
        It's need to design a spec provider
    :param jobName:
    :param taskIndex:
    :return:
    """
    cluster_spec = {
    "worker": [
        "localhost:2222",
        "localhost:2223",
    ],
    "ps": [
        "localhost:2224",
        "localhost:2225"
    ]}
    # Create a cluster from the parameter server and worker hosts.
    cluster = tf.train.ClusterSpec(cluster_spec)

    # Create and start a server for the local task.
    server = tf.train.Server(cluster, jobName, taskIndex)

    return cluster, server


def lrmodel(x, label):
    """
        a simple full connected feedforward function model
    :param x:
    :param label:
    :return:
    """
    IMAGE_PIXELS = 20
    hidden_units = 200
    # Variables of the hidden layer
    hid_w = tf.Variable(tf.truncated_normal([IMAGE_PIXELS * IMAGE_PIXELS, hidden_units],
                                            stddev=1.0 / IMAGE_PIXELS), name="hid_w")
    hid_b = tf.Variable(tf.zeros([hidden_units]), name="hid_b")
    tf.summary.histogram("hidden_weights", hid_w)

    # Variables of the softmax layer
    sm_w = tf.Variable(tf.truncated_normal([hidden_units, 10],
                                           stddev=1.0 / math.sqrt(hidden_units)), name="sm_w")
    sm_b = tf.Variable(tf.zeros([10]), name="sm_b")

    hid_lin = tf.nn.xw_plus_b(x, hid_w, hid_b)
    hid = tf.nn.relu(hid_lin)

    y = tf.nn.softmax(tf.nn.xw_plus_b(hid, sm_w, sm_b))

    loss = -tf.reduce_sum(label * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

    trainOp = tf.train.AdamOptimizer(0.001).minimize(loss)
    return y, loss, trainOp



def tff(model, getClusterAndServer, getlogger):
    """

    :param model:
    :param getClusterAndServer:
    :param getlogger:
    :return:
    """
    def getIndex():
        """
            use a index system to decide worker index
        :return:
        """
        time.sleep(random.random()*0.1)
        api = "http://127.0.0.1:5000/getindex"
        result = eval(requests.get(api).text)
        return int(result["index"])

    def runWorker(index, dataX, dataY, logger):
        jobName = 'worker'
        cluster, server = getClusterAndServer(jobName, index)
        with tf.device(tf.train.replica_device_setter(
                worker_device="/job:worker/task:%d" % index,
                cluster=cluster)):
            x = tf.placeholder(dtype=tf.float32, shape=[None, 400])
            label = tf.placeholder(dtype=tf.float32, shape=[None, 10])
            pred, loss, trainOp = model(x, label)

        init_op = tf.global_variables_initializer()
        sv = tf.train.Supervisor(is_chief=(index == 0),
                                 init_op=init_op,
                                 summary_op=None,
                                 summary_writer=None,
                                 stop_grace_secs=300)
        with sv.managed_session(server.target) as sess:

            for i in range(100):
                _, l = sess.run([trainOp, loss], feed_dict={x: dataX, label: dataY})
                logger.info("workder index %s, iter %s, loss is %s" % (index, i, l))

        logger.info("{0} stopping supervisor".format(datetime.datetime.now().isoformat()))
        sv.stop()

    def runPs(index, logger):
        jobName = 'ps'
        cluster, server = getClusterAndServer(jobName, index)

        logger.info("start server %s" % index)
        server.join()

    def f(iter):
        logger = getlogger()
        data = []
        for item in iter:
            data.append(item)

        data = np.vstack(data)
        logger.info("data shape is %s %s" % (data.shape[0], data.shape[1]))

        dataX = data[:,:400]
        dataY = data[:,400:]
        index = getIndex()

        p = multiprocessing.Process(target=runPs, args=(index, logger,))
        p.start()

        runWorker(index, dataX, dataY, logger)

    return f


def main():
    """
        a simple pyspark + tensorflow + mnist for local distributed test

        Use a index restful api to provide different index, and all executor both
        initialize ps and worker.


    :return:
    """
    # TODO: 1. Cluster spec provider. 2. How to decide whether executor should init ps or not
    # spark = SparkSession.builder.appName("test").master("local").getOrCreate()
    # conf = SparkConf().setAppName("test").setMaster("spark://namenode01:7077") \
    #     .set("spark.shuffle.service.enabled", "false").set("spark.dynamicAllocation.enabled", "false")
        #.set('spark.executor.instances', '1').set('spark.executor.cores', '1')\
        #.set('spark.executor.memory', '512m').set('spark.driver.memory', '512m')\
    conf = SparkConf().setAppName("test").setMaster("local[2]") \
        .set("spark.shuffle.service.enabled", "false").set("spark.dynamicAllocation.enabled", "false")

    sc = SparkContext(conf=conf)
    dataX, dataY = getMnist()
    data = np.split(np.hstack([dataX, dataY]), dataX.shape[0], axis=0)

    rdd = sc.parallelize(data, 2)
    rdd.foreachPartition(tff(lrmodel, getClusterAndServer, getLogger))


if __name__ == '__main__':
    main()