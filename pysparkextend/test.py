
from spark_sklearn.util import createLocalSparkSession
from pysparkextend.tfmodel import TFNeuralNetwork
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


from util.data import getDatasetMinist

def testLr():
    spark = createLocalSparkSession()
    df = getDatasetMinist(spark)
    train, test = df.randomSplit([0.9, 0.1], seed=12345)

    lr = TFNeuralNetwork()
    model = lr.fit(train, {0.01:0.01, 10:10})
    pred = model.transform(test)
    pred.show()

def testCvWithLr():
    spark = createLocalSparkSession()
    df = getDatasetMinist(spark)
    train, test = df.randomSplit([0.9, 0.1], seed=12345)

    lr = TFNeuralNetwork()
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.lr, [0.1, 0.01]) \
        .addGrid(lr.maxIter, [10]) \
        .build()

    tvs = TrainValidationSplit(estimator=lr,
                               estimatorParamMaps=paramGrid,
                               evaluator=MulticlassClassificationEvaluator(),
                               # 80% of the data will be used for training, 20% for validation.
                               trainRatio=0.8)

    model = tvs.fit(train)
    pred = model.transform(test)
    pred.show()


if __name__ == '__main__':
    testLr()