
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import graph_util

from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, is_classifier, clone
from distributed.pysparkTry import lrmodel
from util.data import getMnist
from spark_sklearn.grid_search import GridSearchCV as spGridSearchCV
from pyspark import SparkContext, SparkConf
from sklearn.metrics import accuracy_score
from pyspark.serializers import CloudPickleSerializer

class DisLRModel(BaseEstimator):
    """
        The model is not created in the __init__ for serializing with pyspark

        The graph is not in memory for serializing problem.
        Instead, I saved meta_graph_def and graph_def(constant)
    """
    def __init__(self, inputSize=400, labelSize=10, lr=None, iters=None):

        self.inputSize = inputSize
        self.labelSize = labelSize
        self.lr = lr
        self.iters = iters
        self.exmg = None
        self.bytes = None

    def createGraph(self):
        """
            put the node into graph that init before this function
        :return:
        """
        # with self.g.as_default():
        with tf.variable_scope("lrmodel"):
            input = tf.placeholder(dtype=tf.float32, shape=[None, self.inputSize], name="input")
            label = tf.placeholder(dtype=tf.int32, shape=[None, ], name="label")
            learningRate = tf.placeholder(dtype=tf.float32, name="lr")

            onehotlabel = tf.one_hot(label, self.labelSize, name="onehotlabel")
            pred, loss, globalStep, opt = lrmodel(input, onehotlabel)
            trainOp = tf.train.AdamOptimizer(learningRate).minimize(loss, global_step=globalStep, name="trainOp")

            inits = tf.global_variables_initializer()

    def recoverGraph(self, graph):
        """
            recover data and meta_graph_def into graph initialized before this function
        :param graph:
        :return:
        """
        graph_def = graph.as_graph_def()
        graph_def.ParseFromString(self.bytes)
        # for n in graph_def.node:
        #     print(n.name)
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        tf.train.import_meta_graph(self.exmg)
        tf.import_graph_def(graph_def)

    def getOpFromGraph(self, graph, recover=False):
        """
            get operator and tensor from graph
        :param graph:
        :param recover:
        :return:
        """
        input = graph.get_tensor_by_name('lrmodel/input:0')
        label = graph.get_tensor_by_name('lrmodel/label:0')
        learningRate = graph.get_tensor_by_name('lrmodel/lr:0')
        pred = graph.get_tensor_by_name('lrmodel/pred:0')
        loss = graph.get_tensor_by_name('lrmodel/loss:0')
        # globalStep = graph.get_tensor_by_name('lrmodel/globalStep:0')
        globalStep = tf.contrib.framework.get_or_create_global_step()
        trainOp = graph.get_operation_by_name('lrmodel/trainOp')
        if recover:
            graph_def = graph.as_graph_def()
            valuenames = [n.name + ":0" for n in graph_def.node]
            vs = graph.get_collection('variables')
            inits = [tf.assign(v, graph.get_tensor_by_name("import/" + v.name)) for v in vs if v.name in valuenames]
        else:
            inits = tf.global_variables_initializer()
        return input, label, learningRate, pred, loss, globalStep, trainOp, inits

    def saveGraph(self, sess, graph):
        """
            Serialize the graph into meta_graph_def and graph_def (constant)
        :param sess:
        :param graph:
        :return:
        """
        graph_def = graph.as_graph_def()

        names = [n.name for n in graph_def.node if n.op == "VariableV2" or n.op == "Placeholder"]
        output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, names)
        self.bytes = output_graph_def.SerializeToString()
        self.exmg = tf.train.export_meta_graph(graph=graph)

    def fit(self, dataX, dataY):
        if self.lr and self.iters:
            print(" lr is %s, iters is %s" % (self.lr, self.iters))

            with tf.Graph().as_default() as graph:
                if self.exmg is None or self.bytes is None:
                    self.createGraph()
                    input, label, learningRate, pred, loss, globalStep, trainOp, inits = self.getOpFromGraph(graph, recover=False)
                else:
                    print("recovering graph")
                    self.recoverGraph(graph)
                    input, label, learningRate, pred, loss, globalStep, trainOp, inits = self.getOpFromGraph(graph, recover=True)

                with tf.Session() as sess:
                    sess.run(inits)
                    for i in range(self.iters):
                        _, l = sess.run([trainOp, loss], feed_dict={input: dataX,
                                                                    label: dataY,
                                                                    learningRate: self.lr})
                        print("loss is %s " % l)
                    self.saveGraph(sess, graph)

    def predict(self, dataX):
        with tf.Graph().as_default() as graph:
            print("recovering graph")
            self.recoverGraph(graph)
            input, label, learningRate, pred, loss, globalStep, trainOp, inits = self.getOpFromGraph(graph, recover=True)
            with tf.Session() as sess:
                sess.run(inits)
                p = pred.eval(session=sess, feed_dict={input: dataX})
            if len(p.shape) == 2:
                return np.argmax(p, axis=1)
            return np.argmax(p)

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def get_params(self, deep=True):
        out = {'lr': None, 'iters': None}
        return out



def tensorflowGridSearch():
    """
        Wrap tensorflow so that it can be used in sklearn GridsearchCV
    :return:
    """
    dataX, dataY = getMnist()
    dataX = dataX.astype(np.float32)
    dataY = np.argmax(dataY, axis=1).astype(np.int32)

    tuned_parameters = [{'lr': [1e-1, 1e-2],
                        'iters': [10, 20]}]
    scores = ['precision', 'recall']

    model = DisLRModel(400, 10, 0.01, 10)
    clf = GridSearchCV(model, param_grid=tuned_parameters, cv=2,
                       scoring='%s_macro' % "precision")
    clf.fit(dataX, dataY)

    # test whether the model could be serialized
    cp = CloudPickleSerializer()
    cp.dumps(model)


def sparklearnTensorflowGridSearch():
    """
        Wrap tensorflow so that it can be used in spark-sklearn GridsearchCV
    :return:
    """
    conf = SparkConf().setAppName("test").setMaster("local[2]") \
        .set("spark.shuffle.service.enabled", "false").set("spark.dynamicAllocation.enabled", "false")

    sc = SparkContext(conf=conf)

    dataX, dataY = getMnist()
    dataY = np.argmax(dataY, axis=1)

    tuned_parameters = {'lr': [1e-1, 1e-2],
                        'iters': [10, 20]}
    model = DisLRModel(400, 10, 0.01, 10)

    clf = spGridSearchCV(sc, model, tuned_parameters, cv=2)
    clf.fit(dataX, dataY)




if __name__ == '__main__':
    tensorflowGridSearch()