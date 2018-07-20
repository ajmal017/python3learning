
import sys
import h2o
import time
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from pysparkling import *
from pyspark.sql.session import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import LogisticRegression


def main():
    spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
    sc = spark.sparkContext
    training = spark.createDataFrame([
        (1.218, 1.0, Vectors.dense(1.560, -0.605, '1')),
        (2.949, 0.0, Vectors.dense(0.346, 2.158, '2')),
        (3.627, 0.0, Vectors.dense(1.380, 0.231, '2')),
        (0.273, 1.0, Vectors.dense(0.520, 1.151, '2')),
        (4.199, 0.0, Vectors.dense(0.795, -0.226, '1'))], ["label", "censor", "features"])

    labelIndexer = StringIndexer(inputCol="label", outputCol="indexedLabel").fit(training)
    # Automatically identify categorical features, and index them.
    # We specify maxCategories so features with > 4 distinct values are treated as continuous.
    featureIndexer = \
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(training)

    featureIndexer.transform(training).show()
    lr = LogisticRegression(featuresCol='features', labelCol='censor')
    model = lr.fit(training)
    model.transform(training)
    # hc = H2OContext.getOrCreate(spark)
    # h20frame = hc.as_h2o_frame(training, framename="training")
    # h20frame.summary()
    #
    glm_logistic = H2OGeneralizedLinearEstimator(family="binomial")
    glm_logistic.train(x=['features0','features1'], y='censor', training_frame=h20frame)
    # prediction = glm_logistic.predict(h20frame)
    # prediction.summary()

def get_stdout(model, **kwargs):
    from io import StringIO
    stringio = StringIO()
    sys.stdout = stringio
    model.train(**kwargs)
    # print('get stdout')
    # print(stringio.read())


if __name__ == '__main__':
    main()

