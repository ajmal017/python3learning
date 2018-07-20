import numpy as np
import time
import tensorflow as tf
import datetime
import multiprocessing

from tensorflow.python.framework import graph_util

from pysparkextend.extendutil import getClusterAndServer, getLogger, getIndex
from pyspark import SparkConf, SparkContext, keyword_only
from pyspark.ml.param.shared import HasFeaturesCol, HasLabelCol, HasProbabilityCol, HasRawPredictionCol, HasMaxIter
from pyspark.sql import Row
from pyspark.sql.types import StructField, LongType, DoubleType
from pyspark.ml.param import *
from pyspark.ml.base import Transformer

from distributed.pysparkTry import lrmodel


def parallelFunc(model, maxGlobalStep,
                    getClusterAndServer, getlogger,
                    isPredict=False):
    """
        A demo parallel function for create a tensorflow distributed ps and worker, and running a tensorflow wrapped model

    :param model:
    :param getClusterAndServer:
    :param getlogger:
    :return:
    """
    # TODO: add synchronized distributed learning

    def createDoneQueue(workers, i):
        """Queue used to signal death for i'th ps shard. Intended to have
        all workers enqueue an item onto it to signal doneness."""

        with tf.device("/job:ps/task:%d" % (i)):
            return tf.FIFOQueue(workers, tf.int32, shared_name="done_queue" +
                                                               str(i))

    def createDoneQueues(cluster):
        return [createDoneQueue(cluster.num_tasks('worker'), i) for i in range(cluster.num_tasks('ps'))]

    def runWorker(index, dataX, dataY, logger, result):
        jobName = 'worker'
        isChief = index == 0
        logger.info("index is %s" % index)
        cluster, server = getClusterAndServer(jobName, index)

        with tf.Graph().as_default() as graph:
            with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:worker/task:%d" % index,
                    cluster=cluster)):

                inits, globalStep = model.build(graph, isChief)

                uninitedOp = tf.report_uninitialized_variables()

            queues = createDoneQueues(cluster)
            enqueueOps = [q.enqueue(1) for q in queues]

            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=False,
                device_filters=["/job:ps", "/job:worker/task:%d" % index])

            sv = tf.train.Supervisor(is_chief=isChief,
                                     init_op=inits,
                                     summary_op=None,
                                     summary_writer=None,
                                     stop_grace_secs=300,
                                     global_step=globalStep)

            with sv.prepare_or_wait_for_session(server.target,
                                                config=sess_config) as sess:

                logger.info("worker %d: is initialized, start working" % index)
                # wait for parameter server variables to be initialized
                uniVarList = sess.run(uninitedOp)
                while (len(uniVarList) > 0):
                    logger.info("worker %d: ps uninitialized, sleeping" % index)
                    time.sleep(1)
                    uniVarList = sess.run(uninitedOp)

                if not isPredict:
                    # fit model
                    # TODO: add epoch and mini batch size
                    epoch = 1
                    for i in range(epoch):
                        l, step = model.fit(sess, dataX, dataY)
                        logger.info("workder index %s, loss is %s, global step is %s" % (index, l, step))

                    # wait other worker
                    while step < maxGlobalStep - 1:
                        time.sleep(1e-1)
                        step = sess.run(globalStep)
                    graph_def = graph.as_graph_def()

                    names = [n.name for n in graph_def.node if n.op in ["Variable", "VariableV2", "AutoReloadVariable"]]

                    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, names)
                    bytes = output_graph_def.SerializeToString()
                    exmg = tf.train.export_meta_graph(graph=graph)

                    if isChief:
                        result.append((exmg, bytes))

                else:
                    preds = model.predict(sess, dataX)
                    result += np.split(preds.flatten(), indices_or_sections=preds.shape[0])

                for op in enqueueOps:
                    sess.run(op)

            sv.stop()
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

    def f(iter):
        logger = getlogger()
        rows = []
        labels = []
        for row in iter:
            rows.append(row[0].array)
            if not isPredict:
                labels.append(row[1])

        dataX = np.array(rows)
        dataY = np.array(labels) if not isPredict else None

        logger.info("data shape is %s %s" % (dataX.shape[0], dataX.shape[1]))

        index = getIndex()

        p = multiprocessing.Process(target=runPs, args=(index, logger,))
        p.start()

        result = multiprocessing.Manager().list()
        p2 = multiprocessing.Process(target=runWorker, args=(index, dataX, dataY, logger, result))
        p2.start()
        p2.join()
        result = list(result)
        return result if len(result) > 0 else []

    return f

class TFModel(Transformer):
    """
        Based tensorflow for pyspark wrapped model
    """
    def __init__(self, tfclassifier, metaAndBytes, getClusterAndServer, getLogger):
        self.tfclassifier = tfclassifier
        self.metaAndBytes = metaAndBytes
        self._getClusterAndServer = getClusterAndServer
        self._getLogger = getLogger
        self._copyParamsMap()

    def _copyParamsMap(self):
        self._params = self.tfclassifier.getParams()
        self._paramMap = self.tfclassifier.getParamsMap()
        self._defaultParamMap = self.tfclassifier.getDefaultParamMap()

class TFClassifier(Params):
    """
        Based tensorflow for pyspark wrapped classifier
    """
    def getParams(self):
        return self._params

    def getParamsMap(self):
        return self._paramMap

    def getDefaultParamMap(self):
        return self._defaultParamMap

class HasLr(Params):
    """
        param learning rate: (>= 0).
    """

    lr = Param(Params._dummy(), "lr", "learning rate (>= 0).", typeConverter=TypeConverters.toFloat)

    def __init__(self):
        super(HasLr, self).__init__()

    def setLr(self, value):
        """
        Sets the value of :py:attr:`lr`.
        """
        return self._set(lr=value)

    def getLr(self):
        """
        Gets the value of lr or its default value.
        """
        return self.getOrDefault(self.lr)

class HasPartitions(Params):
    partitions = Param(Params._dummy(), "partitions", "partitions (>= 0).", typeConverter=TypeConverters.toInt)

    def __init__(self):
        super(HasPartitions, self).__init__()

    def setPartitions(self, value):
        """
        Sets the value of :py:attr:`partitions`.
        """
        return self._set(partitions=value)

    def getPartitions(self):
        """
        Gets the value of partitions or its default value.
        """
        return self.getOrDefault(self.partitions)


class TFNeuralNetwork(TFClassifier, HasFeaturesCol, HasLabelCol, HasProbabilityCol, HasRawPredictionCol,
                        HasLr, HasMaxIter, HasPartitions):
    """
        A simple One Hidden Layer tensorflow for pyspark classifier
    """
    @keyword_only
    def __init__(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                 probabilityCol="probability", rawPredictionCol="rawPrediction",
                 lr=0.01, maxIter=10, partitions=2):
        super(TFNeuralNetwork, self).__init__()
        self._setDefault(lr=0.01, maxIter=10, partitions=2)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

        self.model = TFNeuralNetworkSimple
        self._getClusterAndServer = getClusterAndServer
        self._getLogger = getLogger

    @keyword_only
    def setParams(self, featuresCol="features", labelCol="label", predictionCol="prediction",
                 probabilityCol="probability", rawPredictionCol="rawPrediction",
                 lr=0.01, maxIter=2, partitions=2):
        kwargs = self._input_kwargs
        return self._set(**kwargs)


    def fit(self, dataset, params=None):
        """
            Fits a model to the input dataset with optional parameters.

        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :param params: an optional param map that overrides embedded params. If a list/tuple of
                       param maps is given, this calls fit on each param map and returns a list of
                       models.
        :returns: fitted model(s)
        """
        if params is None:
            params = dict()
        if isinstance(params, (list, tuple)):
            return [self.fit(dataset, paramMap) for paramMap in params]
        elif isinstance(params, dict):
            # paramMap is params dict for self._fit
            # self.params contains all param (name) set in __init__
            # self._defaultParamMap own all params default value
            paramMap = {}
            for p in self.params:
                defaultValue = self._defaultParamMap[p]
                if defaultValue in params:
                    paramMap[p.name] = params[defaultValue]
                elif p in params:
                    paramMap[p.name] = params[p]
                elif p in self._paramMap:
                    paramMap[p.name] = self._paramMap[p]
                else:
                    paramMap[p.name] = defaultValue
            return self._fit(dataset, **paramMap)
            # return None

        else:
            raise ValueError("Params must be either a param map or a list/tuple of param maps, "
                             "but got %s." % type(params))


    def _fit(self, dataset, featuresCol="features", labelCol="label", predictionCol="prediction",
                 probabilityCol="probability", rawPredictionCol="rawPrediction",
                 lr=0.01, maxIter=2, partitions=2):
        model = self.model(400, 10, lr=lr, maxIter=maxIter, partitions=partitions)
        maxGlobalStep = maxIter * partitions
        getClusterAndServer = self._getClusterAndServer()
        getLogger = self._getLogger

        metaAndBytes = self._fitTf(dataset, featuresCol, labelCol, partitions, parallelFunc(model,
                                                       maxGlobalStep, getClusterAndServer, getLogger))
        return TFNeuralNetworkModel(self, metaAndBytes, self._getClusterAndServer, self._getLogger)

    def _fitTf(self, dataset, featuresCol, labelCol, partitions, mapfun):
        rdd = dataset.select(featuresCol, labelCol).rdd
        rdd = rdd.repartition(partitions)
        metaAndBytes = rdd.mapPartitions(mapfun).collect()[0]
        return metaAndBytes


class TFNeuralNetworkModel(TFModel):
    """
        A simple One Hidden Layer tensorflow for pyspark model
    """
    def transform(self, dataset, params=None):
        """
        Transforms the input dataset with optional parameters.

        :param dataset: input dataset, which is an instance of :py:class:`pyspark.sql.DataFrame`
        :param params: an optional param map that overrides embedded params.
        :returns: transformed dataset
        """
        print("predicting %s" % str(params))
        if params is None:
            params = dict()
        if isinstance(params, (list, tuple)):
            return [self.transform(dataset, paramMap) for paramMap in params]
        elif isinstance(params, dict):
            # paramMap is params dict for self._fit
            # self.params contains all param (name) set in __init__
            # self._defaultParamMap own all params default value
            paramMap = {}
            for p in self.params:
                defaultValue = self._defaultParamMap[p]
                if defaultValue in params:
                    paramMap[p.name] = params[defaultValue]
                elif p in params:
                    paramMap[p.name] = params[p]
                elif p in self._paramMap:
                    paramMap[p.name] = self._paramMap[p]
                else:
                    paramMap[p.name] = defaultValue
            return self._transformTf(dataset, **paramMap)

        else:
            raise ValueError("Params must be either a param map or a list/tuple of param maps, "
                             "but got %s." % type(params))


    def _transformTf(self, dataset, featuresCol="features", labelCol="label", predictionCol="prediction",
                    probabilityCol="probability", rawPredictionCol="rawPrediction",
                    lr=0.01, maxIter=2, partitions=2):

        model = self.tfclassifier.model(400, 10, lr=lr, maxIter=maxIter, partitions=partitions, metaAndBytes=self.metaAndBytes)
        maxGlobalStep = maxIter * partitions
        getClusterAndServer = self._getClusterAndServer()
        getLogger = self._getLogger
        dataset = dataset.repartition(partitions)
        rdd = dataset.select(featuresCol).rdd
        results = rdd.mapPartitions(parallelFunc(model, maxGlobalStep, getClusterAndServer, getLogger, isPredict=True))

        def combine(d_r):
            return d_r[0] + Row(pred=float(d_r[1][0]))

        resultRdd = dataset.rdd.zip(results).map(combine)
        resultDf = dataset.sql_ctx.createDataFrame(resultRdd, dataset.schema.add(predictionCol, DoubleType())).cache()
        resultDf.count()
        return resultDf


class TFNeuralNetworkSimple():
    """
        A tensorflow one hidden layer model wrapper

    """
    def __init__(self, inputSize=None, labelSize=None, lr=0.01, maxIter=10, partitions=2, metaAndBytes=None):
        self.inputSize = inputSize
        self.labelSize = labelSize
        self.lr = lr
        self.maxIter = maxIter
        self.partitions = partitions
        self.exmg = metaAndBytes[0] if metaAndBytes is not None else None
        self.bytes = metaAndBytes[1] if metaAndBytes is not None else None


    def build(self, graph, isChief):
        if self.exmg is None or self.bytes is None or not isChief:
            self.createGraph()
        else:
            print("recovering graph!!!  %s" % isChief)
            self.recoverGraph(graph)
        return self.inits, self.globalStep

    def createGraph(self):
        """
            put the node into graph that init before this function
        :return:
        """
        # with self.g.as_default():
        with tf.variable_scope("lrmodel"):
            self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.inputSize], name="input")
            self.label = tf.placeholder(dtype=tf.int32, shape=[None, ], name="label")
            self.learningRate = tf.placeholder(dtype=tf.float32, name="lr")

            self.onehotlabel = tf.one_hot(self.label, self.labelSize, name="onehotlabel")
            self.pred, self.loss, self.globalStep, self.opt = lrmodel(self.input, self.onehotlabel)

            self.opt = tf.train.SyncReplicasOptimizer(self.opt, replicas_to_aggregate=self.partitions,
                                                 total_num_replicas=self.partitions, name="opt2")

            self.trainOp = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss, global_step=self.globalStep, name="trainOp")
            self.inits = tf.global_variables_initializer()

    def recoverGraph(self, graph):
        """
            recover data and meta_graph_def into graph initialized before this function
        :param graph:
        :return:
        """

        graph_def = graph.as_graph_def()
        graph_def.ParseFromString(self.bytes)

        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.train.import_meta_graph(self.exmg)
        tf.import_graph_def(graph_def)

        self.getOpFromGraph(graph, True)

    def getOpFromGraph(self, graph, recover=False):
        """
            get operator and tensor from graph
        :param graph:
        :param recover:
        :return:
        """
        self.input = graph.get_tensor_by_name('lrmodel/input:0')
        self.label = graph.get_tensor_by_name('lrmodel/label:0')
        self.learningRate = graph.get_tensor_by_name('lrmodel/lr:0')
        self.pred = graph.get_tensor_by_name('lrmodel/pred:0')
        self.loss = graph.get_tensor_by_name('lrmodel/loss:0')
        self.globalStep = tf.contrib.framework.get_or_create_global_step()
        self.trainOp = graph.get_operation_by_name('lrmodel/trainOp')
        if recover:
            graph_def = graph.as_graph_def()
            valuenames = [n.name + ":0" for n in graph_def.node]
            vs = graph.get_collection('variables')
            self.inits = [tf.assign(v, graph.get_tensor_by_name("import/" + v.name)) for v in vs if v.name in valuenames]
        else:
            self.inits = tf.global_variables_initializer()


    def fit(self, sess, dataX, dataY):
        """
            fit data
        :param sess:
        :param dataX:
        :param dataY:
        :return:  loss , global step
        """
        print("lr is %s, iters is %s" % (self.lr, self.maxIter))

        for i in range(self.maxIter):
            _, l, s = sess.run([self.trainOp, self.loss, self.globalStep], feed_dict={self.input: dataX,
                                                        self.label: dataY,
                                                        self.learningRate: self.lr})
        return l, s

    def predict(self, sess, dataX):
        """
            prediction are numbers instead of one hot mapping
        :param sess:
        :param dataX:
        :return:
        """
        p = self.pred.eval(session=sess, feed_dict={self.input: dataX})
        return np.argmax(p, axis=1)

