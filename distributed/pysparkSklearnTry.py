
import numpy as np
import pyspark
import sklearn
from sklearn import linear_model
from pyspark import SparkConf, SparkContext
from util.data import getMnist, getDatasetMinist
from distributed.pysparkTry import getLogger
from pyspark.sql.functions import udf
from pyspark.ml.param import Param, Params
from pyspark.ml.linalg import Vectors, Matrices, MatrixUDT
from pyspark.ml.regression import GBTRegressor

from spark_sklearn.util import createLocalSparkSession


class Base():
    def _fit(self, x):
        return x**2

class Test():
    pass

class Impl(Base, Test):
    def fit(self, x):
        return super(Impl, self)._fit(x)




def map(getLogger):
    """
        For spark rdd
    :param getLogger:
    :return:
    """
    def f(m):
        logger = getLogger()
        data = []
        iter = list(m)[0][1]
        for item in iter:
            data.append(item)

        data = np.vstack(data)
        logger.info("data shape is %s %s" % (data.shape[0], data.shape[1]))

        dataX = data[:, :400]
        dataY = data[:, 400:]
        dataY = np.argmax(dataY, axis=1)

        model = linear_model.LogisticRegression()
        model.fit(dataX, dataY)
        return [model]

    return f

def mapPredict(getLogger, modelCollect):
    def f(iter):
        logger = getLogger()
        data = []
        for item in iter:
            data.append(item)

        data = np.vstack(data)
        logger.info("data shape is %s %s" % (data.shape[0], data.shape[1]))

        result = np.zeros((data.shape[0], 10))
        for model in modelCollect:
            pred = model.predict(data)
            for i, p in enumerate(pred):
                result[i, p] += 1
        result = np.argmax(result, axis=1)
        return np.split(result, result.shape[0])
    return f

class RDDLRSklModel(object):
    """
        Using Logistic model in pyspark's rdd
    """
    def __init__(self, getLogger):
        self.modelRdd = None
        self.getLogger = getLogger

    def fit(self, dataset):
        dataset = dataset.map(lambda d: (np.random.randint(0,2), d)).groupByKey()
        self.modelRdd = dataset.mapPartitions(map(self.getLogger)).collect()

    def predict(self, sc, dataset):
        sc.broadcast(self.modelRdd)

        result = dataset.mapPartitions(mapPredict(self.getLogger, self.modelRdd)).collect()
        result = np.array(result).flatten()
        return result



class SKLModel(pyspark.ml.Estimator):
    """
        fit spark dataframe as input
    """
    def __init__(self, sklearnEstimator=None, keyCols=["key"], xCol="features",
                 outputCol="output", yCol=None, estimatorType=None):
        if sklearnEstimator is None:
            raise ValueError("sklearnEstimator should be specified")
        if not isinstance(sklearnEstimator, sklearn.base.BaseEstimator):
            raise ValueError("sklearnEstimator should be an sklearn.base.BaseEstimator")
        # if len(keyCols) == 0:
        #     raise ValueError("keyCols should not be empty")
        if "estimator" in keyCols + [xCol, yCol]:
            raise ValueError("keyCols should not contain a column named \"estimator\"")

        # The superclass expects Param attributes to already be set, so we only init it after
        # doing so.
        for paramName, paramSpec in SKLModel._paramSpecs.items():
            setattr(self, paramName, Param(Params._dummy(), paramName, paramSpec["doc"]))
        super(SKLModel, self).__init__()
        self._setDefault(**{paramName: paramSpec["default"]
                            for paramName, paramSpec in SKLModel._paramSpecs.items()
                            if "default" in paramSpec})
        kwargs = SKLModel._inferredParams(sklearnEstimator, self._input_kwargs)
        self._set(**kwargs)

        self._verifyEstimatorType()

    @staticmethod
    def _inferredParams(estimator, inputParams):
        if "estimatorType" in inputParams:
            return inputParams
        if "yCol" in inputParams:
            inputParams["estimatorType"] = "predictor"
        elif hasattr(estimator, "fit_predict"):
            inputParams["estimatorType"] = "clusterer"
        else:
            inputParams["estimatorType"] = "transformer"
        return inputParams

def main():
    """
        a simple distributed multi model execution

        for input rdd, it's must be shuffle to avoid label imbalance in different partition.

        I have used a simple method that was not confirmed to be balanced partition.

        I thought it was important for user to do data sampling before algorithm.
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

    index = np.arange(dataX.shape[0])
    # np.random.shuffle(index)
    # dataX = dataX[index]
    # dataY = dataY[index]

    data = np.split(np.hstack([dataX, dataY]), dataX.shape[0], axis=0)

    rdd = sc.parallelize(data, 2)
    rdd2 = sc.parallelize(dataX, 2)

    sklmodel = RDDLRSklModel(getLogger)
    sklmodel.fit(rdd)
    result = sklmodel.predict(sc, rdd2)

    # result2 = simpleLR()
    dataY = np.argmax(dataY, axis=1)
    print(result)
    print(dataY)
    print(np.sum(result == dataY))
    sc.stop()

def simpleLR():
    dataX, dataY = getMnist()
    dataY = np.argmax(dataY, axis=1)

    model = linear_model.LogisticRegression()
    model.fit(dataX, dataY)
    pred = model.predict(dataX)
    print(np.sum(pred == dataY))


def udfTest():
    # conf = SparkConf().setAppName("test").setMaster("local[2]") \
    #     .set("spark.shuffle.service.enabled", "false").set("spark.dynamicAllocation.enabled", "false")
    #
    # sc = SparkContext(conf=conf)
    spark = createLocalSparkSession()
    dataX, dataY = getMnist()

    def sklmodelPredict(model):
        def f(vec):
            p = model.predict(np.array(vec.values.data).reshape(1, -1))
            return int(p)
        return f



    df = spark.createDataFrame([(user,
                                 Vectors.dense([i, i ** 2, i ** 3]),
                                0.0 + user + i + 2 * i ** 2 + 3 * i ** 3)
                                for user in range(3) for i in range(5)])
    df = df.toDF("key", "features", "y")
    pd = df.select('features', 'y').toPandas()
    dataX = np.vstack(pd['features'].apply(lambda v: v.toArray()))
    dataY = pd['y'].values.reshape(-1, 1)
    model = linear_model.LinearRegression()
    model.fit(dataX, dataY)

    ufun = udf(sklmodelPredict(model))

    df.withColumn("pred", ufun("features")).show()




if __name__ == '__main__':
    udfTest()