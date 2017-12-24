
import numpy as np
import tensorflow as tf
import multiprocessing
import time
import random
import requests
import datetime


from util.data import getMnist

from tensorflow.python.framework import graph_util

from distributed.gridsearch import DisLRModel
from distributed.pysparkTry import getLogger, getClusterAndServer

from pyspark import SparkConf, SparkContext, keyword_only
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.linalg import Vectors
from spark_sklearn.util import createLocalSparkSession
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasProbabilityCol, HasRawPredictionCol
from pyspark.sql.functions import udf



def getSC():
    conf = SparkConf().setAppName("test").setMaster("local[2]") \
        .set("spark.shuffle.service.enabled", "false").set("spark.dynamicAllocation.enabled", "false")

    sc = SparkContext(conf=conf)
    return sc

def getDatasetMinist(spark):
    dataX, dataY = getMnist()
    dataY = np.argmax(dataY, axis=1)

    index = np.arange(dataX.shape[0])
    np.random.shuffle(index)
    dataX = dataX[index]
    dataY = dataY[index]

    # Vectors.dense([1,2,3]).array

    df = spark.createDataFrame([(Vectors.dense(x.tolist()), int(y))
                                for x, y in zip(dataX, dataY)
                                ])
    df = df.toDF("features", "label")
    return df

def pysparkLR():
    spark = createLocalSparkSession()
    df = getDatasetMinist(spark)

    train, test = df.randomSplit([0.9, 0.1], seed=12345)

    lr = RandomForestClassifier()
    LogisticRegression
    lr.fit()

    paramGrid = ParamGridBuilder() \
        .addGrid(lr.maxDepth, [4, 5]) \
        .addGrid(lr.numTrees, [10, 20]) \
        .build()

    tvs = TrainValidationSplit(estimator=lr,
                               estimatorParamMaps=paramGrid,
                               evaluator=MulticlassClassificationEvaluator(),
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)

    model = tvs.fit(train)

    # Make predictions on test data. model is the model with combination of parameters
    # that performed best.
    model.transform(test) \
        .select("features", "label", "prediction") \
        .show(500)





class TFMultiPercetion(HasFeaturesCol, HasLabelCol, HasProbabilityCol, HasRawPredictionCol):
    """
        Pyspark ml expanding with tensorflow
    """
    @keyword_only
    def __init__(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                 probabilityCol="probability", rawPredictionCol="rawPrediction",
                 featuresSize=400, labelSize=10, lr=0.01, iters = 10, partitions=2,
                 getClusterAndServer=getClusterAndServer, getLogger=getLogger):
        self.featuresCol = featuresCol
        self.labelCol = labelCol
        self.featuresSize = featuresSize
        self.labelSize = labelSize
        self.lr = lr
        self.iters = iters
        self.model = DisLRModel(lr=self.lr, iters=self.iters)
        self.models = None
        self.partitions= partitions
        self.maxGlobalStep = iters * partitions
        self.metaAndBytes = None
        self.getClusterAndServer = getClusterAndServer
        self.getLogger = getLogger

    def fit(self, dataset):
        model = self.model
        maxGlobalStep = self.maxGlobalStep
        getClusterAndServer = self.getClusterAndServer
        getlogger = self.getLogger
        lr = self.lr
        iters = self.iters

        def parallelFitFunc(model, getClusterAndServer, getlogger):
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
                time.sleep(random.random() * 0.1)
                api = "http://127.0.0.1:5000/getindex"
                result = eval(requests.get(api).text)
                return int(result["index"])

            def createDoneQueue(workers, i):
                """Queue used to signal death for i'th ps shard. Intended to have
                all workers enqueue an item onto it to signal doneness."""

                with tf.device("/job:ps/task:%d" % (i)):
                    return tf.FIFOQueue(workers, tf.int32, shared_name="done_queue" +
                                                                       str(i))

            def createDoneQueues(cluster):
                return [createDoneQueue(cluster.num_tasks('worker'), i) for i in range(cluster.num_tasks('ps'))]

            def runWorker(index, dataX, dataY, logger):
                jobName = 'worker'
                isChief = index == 0
                logger.info("index is %s" % index)
                cluster, server = getClusterAndServer(jobName, index)

                with tf.Graph().as_default() as graph:
                    with tf.device(tf.train.replica_device_setter(
                            worker_device="/job:worker/task:%d" % index,
                            cluster=cluster)):

                        if self.metaAndBytes is None:
                            model.createGraph()
                            input, label, learningRate, pred, loss, globalStep, trainOp, inits = model.getOpFromGraph(graph,
                                                                                                         recover=False)
                        else:
                            print("recovering graph")
                            model.recoverGraph(graph)
                            input, label, learningRate, pred, loss, globalStep, trainOp, inits = model.getOpFromGraph(graph,
                                                                                                         recover=True)

                        uninitedOp = tf.report_uninitialized_variables()

                    queues = createDoneQueues(cluster)
                    enqueueOps = [q.enqueue(1) for q in queues]

                    init_op = tf.global_variables_initializer()
                    sv = tf.train.Supervisor(is_chief=(isChief),
                                             init_op=init_op,
                                             summary_op=None,
                                             summary_writer=None,
                                             stop_grace_secs=300,
                                             global_step=globalStep)

                    with sv.managed_session(server.target) as sess:

                        # wait for parameter server variables to be initialized
                        while (len(sess.run(uninitedOp)) > 0):
                            logger.info("worker %d: ps uninitialized, sleeping" % index)
                            time.sleep(1)

                        logger.info("worker %d: is initialized, start working" % index)
                        for i in range(iters):
                            _, l, step = sess.run([trainOp, loss, globalStep], feed_dict={input: dataX,
                                                                                          label: dataY,
                                                                                          learningRate: lr})
                            logger.info("workder index %s, iter %s, loss is %s, global step is %s" % (index, i, l, step))

                        while step < maxGlobalStep-1:
                            logger.info(
                                "global step is %s" % (step))
                            time.sleep(1e-1)
                            step = sess.run(globalStep)
                        graph_def = graph.as_graph_def()

                        names = [n.name for n in graph_def.node if n.op == "VariableV2" or n.op == "Placeholder"]
                        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, names)
                        bytes = output_graph_def.SerializeToString()
                        exmg = tf.train.export_meta_graph(graph=graph)

                        for op in enqueueOps:
                            sess.run(op)

                    logger.info("{0} stopping supervisor".format(datetime.datetime.now().isoformat()))
                    sv.stop()
                    if isChief:
                        return [(bytes, exmg)]
                    return []

            def runWorkerSyn(index, dataX, dataY, logger):
                """
                    A synchronized distributed training method
                :param index:
                :param dataX:
                :param dataY:
                :param logger:
                :return:
                """
                jobName = 'worker'
                is_chief = (index == 0)
                cluster, server = getClusterAndServer(jobName, index)
                with tf.device(tf.train.replica_device_setter(
                        worker_device="/job:worker/task:%d" % index,
                        cluster=cluster)):
                    x = tf.placeholder(dtype=tf.float32, shape=[None, 400])
                    label = tf.placeholder(dtype=tf.float32, shape=[None, 10])
                    pred, loss, globalStep, opt = model(x, label)

                    opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=2,
                                                         total_num_replicas=2)
                    trainOp = opt.minimize(loss, global_step=globalStep)

                    uninitedOp = tf.report_uninitialized_variables()

                queues = createDoneQueues(cluster)
                enqueueOps = [q.enqueue(1) for q in queues]

                if is_chief:
                    chief_queue_runner = opt.get_chief_queue_runner()
                    init_tokens_op = opt.get_init_tokens_op()

                if is_chief:
                    logger.info("Worker %d: Initializing session..." % index)
                else:
                    logger.info("Worker %d: Waiting for session to be initialized..." %
                                index)

                sess_config = tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=False,
                    device_filters=["/job:ps", "/job:worker/task:%d" % index])

                init_op = tf.global_variables_initializer()

                sv = tf.train.Supervisor(is_chief=is_chief,
                                         init_op=init_op,
                                         summary_op=None,
                                         summary_writer=None,
                                         stop_grace_secs=300,
                                         global_step=globalStep)

                with sv.prepare_or_wait_for_session(server.target,
                                                    config=sess_config) as sess:

                    print("Starting chief queue runner and running init_tokens_op")
                    if is_chief:
                        sv.start_queue_runners(sess, [chief_queue_runner])
                        sess.run(init_tokens_op)

                    # wait for parameter server variables to be initialized
                    while (len(sess.run(uninitedOp)) > 0):
                        logger.info("worker %d: ps uninitialized, sleeping" % index)
                        time.sleep(1)

                    step = 0
                    logger.info("worker %d: is initialized, start working" % index)
                    while step < 100:
                        _, l, step = sess.run([trainOp, loss, globalStep], feed_dict={x: dataX, label: dataY})

                        logger.info("workder index %s, loss is %s, global_step is %s" % (index, l, step))

                    for op in enqueueOps:
                        sess.run(op)

                logger.info("{0} stopping supervisor".format(datetime.datetime.now().isoformat()))

            def runPs(index, logger):
                jobName = 'ps'
                cluster, server = getClusterAndServer(jobName, index)

                logger.info("start server %s" % index)

                queue = createDoneQueue(cluster.num_tasks('worker'), index)

                with tf.Session(server.target) as sess:
                    for i in range(cluster.num_tasks('worker')):
                        sess.run(queue.dequeue())
                logger.info("end server %s" % index)

            def runPsSyn(index, logger):
                jobName = 'ps'
                cluster, server = getClusterAndServer(jobName, index)

                logger.info("start server %s" % index)
                server.join()
                logger.info("end server %s" % index)

            def fSyn(iter):
                logger = getlogger()
                data = []
                for item in iter:
                    data.append(item)

                data = np.vstack(data)
                logger.info("data shape is %s %s" % (data.shape[0], data.shape[1]))

                dataX = data[:, :400]
                dataY = data[:, 400:]
                index = getIndex()

                p = multiprocessing.Process(target=runPsSyn, args=(index, logger,))
                p.start()

                runWorkerSyn(index, dataX, dataY, logger)

            def f(iter):
                logger = getlogger()
                rows = []
                labels = []
                for row in iter:
                    rows.append(row[0].array)
                    labels.append(row[1])
                dataX = np.array(rows)
                dataY = np.array(labels)

                logger.info("data shape is %s %s" % (dataX.shape[0], dataX.shape[1]))

                index = getIndex()

                p = multiprocessing.Process(target=runPs, args=(index, logger,))
                p.start()

                return runWorker(index, dataX, dataY, logger)

            return f

        self._fit(dataset, parallelFitFunc(model, getClusterAndServer, getlogger))

    def _fit(self, dataset, mapfun):
        rdd = dataset.select(self.featuresCol, self.labelCol).rdd
        rdd = rdd.repartition(self.partitions)
        self.metaAndBytes = rdd.mapPartitions(mapfun).collect()

    def transform(self, dataset):
        def tfmodelPredict(models):
            def f(vec):
                preds = {}
                for model in models:
                    p = model.predict(np.array(vec.values.data).reshape(1, -1))
                    p = int(p)
                    if p not in preds:
                        preds[p] = 1
                    else:
                        preds[p] += 1
                p, c = max(list(preds.items()), key=lambda k_c:k_c[1])
                return p

            return f

        dataset.select(self.featuresCol).rdd.mapPartitions()

        return


def TFMultiPercetionTest():
    spark = createLocalSparkSession()
    df = getDatasetMinist(spark)

    model = TFMultiPercetion()
    model.fit(df)
    # print(model.metaAndBytes)
    # pred = model.transform(df)
    # pred.show()

if __name__ == '__main__':
    TFMultiPercetionTest()
