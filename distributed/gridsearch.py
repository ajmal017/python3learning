
import tensorflow as tf
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, is_classifier, clone
from distributed.pysparkTry import lrmodel
from util.data import getMnist
from spark_sklearn.grid_search import GridSearchCV as spGridSearchCV
from pyspark import SparkContext, SparkConf
from sklearn.metrics import accuracy_score


class LRModel(BaseEstimator):
    """
        The model is not created in the __init__ for serializing with pyspark
    """
    def __init__(self, inputSize=400, labelSize=10, lr=None, iters=None):

        self.inputSize = inputSize
        self.labelSize = labelSize
        self.lr = lr
        self.iters = iters
        self.input = None

    def createModel(self):
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, self.inputSize])
        self.indices = tf.placeholder(dtype=tf.int32, shape=[None, ])
        self.label = tf.one_hot(self.indices, self.labelSize)
        self.y, self.loss, self.globalStep, self.opt = lrmodel(self.input, self.label)

        self.learningRate = tf.placeholder(dtype=tf.float32)
        self.trainOp = tf.train.AdamOptimizer(self.learningRate).minimize(self.loss)


    def fit(self, dataX, dataY):
        if self.input == None:
            self.createModel()
        if self.lr and self.iters:
            print(" lr is %s, iters is %s" % (self.lr, self.iters))
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())

                for i in range(self.iters):
                    _, l = sess.run([self.trainOp, self.loss], feed_dict={self.input: dataX,
                                                                   self.indices: dataY,
                                                                      self.learningRate: self.lr})
                    print("loss is %s " % l)

    def predict(self, dataX):
        if self.input is None:
            raise Exception("please create model first")
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            pred = self.y.eval(session=sess, feed_dict={self.input: dataX})
        if len(pred.shape) == 2:
            return np.argmax(pred, axis=1)
        return np.argmax(pred)

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

    dataY = np.argmax(dataY, axis=1)

    tuned_parameters = [{'lr': [1e-1, 1e-2],
                        'iters': [10, 20]}]
    scores = ['precision', 'recall']

    model = LRModel(400, 10, 0.01, 10)
    clf = GridSearchCV(model, param_grid=tuned_parameters, cv=2,
                       scoring='%s_macro' % "precision")
    clf.fit(dataX, dataY)


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
    model = LRModel(400, 10, 0.01, 10)

    clf = spGridSearchCV(sc, model, tuned_parameters, cv=2)
    clf.fit(dataX, dataY)




if __name__ == '__main__':
    sparklearnTensorflowGridSearch()