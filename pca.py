from pyspark.sql import SparkSession
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA as PCAml
import mlflow.spark
import sys

# Source data path.
train_set_uri = '.\\data\\USArrests.csv'

# How many components are we looking for?
pca_components = int(sys.argv[1]) if len(sys.argv) > 1 else 2

# Build spark session and include mlflow
spark = SparkSession \
    .builder \
    .appName('pca') \
    .getOrCreate()

# Load the training set.
df = spark \
    .read \
    .format("csv") \
    .option('inferSchema', 'true') \
    .option('header', 'true') \
    .load(train_set_uri)

# Assemble the Feature Vector
assembler = VectorAssembler(
    inputCols=['Murder', 'Assault', 'UrbanPop', 'Rape'],
    outputCol='features')

# Create a scaler to normalize the feature values
scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=True)

# Build PCA Model
pca = PCAml(k=pca_components, inputCol="scaled_features", outputCol="pca")

try:
    with mlflow.start_run():
        ml_pipeline = Pipeline(stages=[assembler, scaler, pca])
        model = ml_pipeline.fit(df)
        # Make predictions
        predictions = model.transform(df)

        mlflow.log_param('pcas', float(pca_components))

        # Shows the result.
        exp_var_list = model.stages[-1].explainedVariance
        i = 0
        for var in exp_var_list:
            mlflow.log_metric(f'pca_expl_var_{i}', float(var))
            i = i + 1

        mlflow.spark.log_model(model, "pca")
except Exception as e:
    print(str(e))

print('Done')
