from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
import mlflow.spark
import sys

spark = SparkSession \
    .builder \
    .config("spark.jars.packages", "org.mlflow:mlflow-spark:1.12.1") \
    .appName('kmean_clustering') \
    .getOrCreate()

mlflow.spark.autolog()

df = spark \
    .read \
    .format("csv") \
    .option('inferSchema', 'true') \
    .option('header', 'true') \
    .load(r"C:\Users\dhamacher\Desktop\BLOB\kmean_test.csv") \
    .cache()

colm = df.columns
feature_columns = colm[3:]
clusters = int(sys.argv[1]) if len(sys.argv) > 1 else 1

# Assemble the Feature Vector
assembler = VectorAssembler(inputCols=feature_columns, outputCol='features')

# df = assembler.transform(df)
# print(df.select('*').first())

# Scale the feature Vector prior to PCA
scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=True)

# Compute summary statistics by fitting the StandardScaler
# scalerModel = scaler.fit(df)

# Normalize each feature to have unit standard deviation.
# df = scalerModel.transform(df)

# Trains a k-means model.
kmeans = KMeans(featuresCol='scaled_features', predictionCol='prediction', k=clusters).setSeed(1)
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

        mlflow.log_param('Silhouette', str(silhouette))

        # Shows the result.
        centers = model.stages[-1].clusterCenters()
        i = 0
        for center in centers:
            mlflow.log_param(f'cluster_{i}', center)
            i = i+1

        mlflow.spark.log_model(model, "coffee_kmeans")
except Exception as e:
    print(str(e))



# predictions.select('YEAR', 'MONTH', 'CARD_ACCOUNT_ID', 'ESPRESSO', 'DRIP', 'COLD_BREW', 'BEANS', 'WARM_FOOD',
#                    'BAKED_GOODS', 'prediction') \
#     .repartition(1) \
#     .write \
#     .mode('overwrite') \
#     .option('headers', 'true') \
#     .format('csv') \
#     .save("C:\\Users\\dhamacher\\Desktop\\BLOB\\kmean_pred.csv")
