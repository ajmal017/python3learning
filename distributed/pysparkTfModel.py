
from util.data import getMnist, getDatasetMinist

from pyspark import SparkConf, SparkContext, keyword_only
from spark_sklearn.util import createLocalSparkSession
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import RandomForestClassifier




def getSC():
    conf = SparkConf().setAppName("test").setMaster("local[2]") \
        .set("spark.shuffle.service.enabled", "false").set("spark.dynamicAllocation.enabled", "false")

    sc = SparkContext(conf=conf)
    return sc


def pysparkLR():
    """
        TrainValidationSplit Test
    :return:
    """
    spark = createLocalSparkSession()
    df = getDatasetMinist(spark)

    train, test = df.randomSplit([0.9, 0.1], seed=12345)

    lr = RandomForestClassifier()

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



if __name__ == '__main__':
    pysparkLR()
