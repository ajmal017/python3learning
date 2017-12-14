
import numpy as np
from sklearn import linear_model
from pyspark import SparkConf, SparkContext
from util.data import getMnist
from distributed.pysparkTry import getLogger

class Base():
    def _fit(self, x):
        return x**2

class Test():
    pass

class Impl(Base, Test):
    def fit(self, x):
        return super(Impl, self)._fit(x)




def map(getLogger):
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

if __name__ == '__main__':
    simpleLR()