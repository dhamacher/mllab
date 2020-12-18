from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import mlflow.spark
import sys

train_set_uri = '.\\data\\USArrests.csv'
k_clusters = int(sys.argv[1]) if len(sys.argv) > 1 else 2


spark = SparkSession \
    .builder \
    .appName('kmeans') \
    .getOrCreate()

# Load the training set
df = spark \
    .read \
    .format("csv") \
    .option('inferSchema', 'true') \
    .option('header', 'true') \
    .load(train_set_uri)

# Get the feature column names
colm = df.columns
feature_cols = colm[1:]

# Assemble the Feature Vector
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol='features')

# Scale the feature Vector prior to using kmean clustering
scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=True)

# Trains a k-means model.
kmeans = KMeans(featuresCol='scaled_features', predictionCol='prediction', k=k_clusters).setSeed(1)


try:
    with mlflow.start_run():
        ml_pipeline = Pipeline(stages=[assembler, scaler, kmeans])
        model = ml_pipeline.fit(df)
        # Make predictions
        predictions = model.transform(df)

        # Evaluate clustering by computing Silhouette score
        evaluator = ClusteringEvaluator()
        silhouette = evaluator.evaluate(predictions)
        # print("Silhouette with squared euclidean distance = " + str(silhouette))

        mlflow.log_metric('silhouette_score', float(silhouette))
        mlflow.log_param('k_clusters', k_clusters)

        # Shows the result.
        centers = model.stages[-1].clusterCenters()
        i = 0
        for center in centers:
            mlflow.log_param(f'cluster_{i}', center)
            i = i+1

        mlflow.spark.log_model(model, "kmeans")
except Exception as e:
    print(str(e))

print('Done')
