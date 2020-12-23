import pyspark.sql.functions as F
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import *
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import PCA as PCAml
from pyspark.ml.linalg import Vectors  # Pre 2.0 pyspark.mllib.linalg

spark = SparkSession.builder.appName('pca').getOrCreate()

df = spark \
    .read \
    .format("csv") \
    .option('inferSchema', 'true') \
    .option('header', 'true') \
    .load(r"C:\Users\dhamacher\Desktop\BLOB\kmean_test.csv") \
    .cache()

# Assemble the Feature Vector
assembler = VectorAssembler(
    inputCols=['ESPRESSO', 'DRIP', 'COLD_BREW', 'BEANS', 'WARM_FOOD', 'BAKED_GOODS'],
    outputCol='features')

df = assembler.transform(df)
# print(df.select('*').first())

# Scale the feature Vector prior to PCA
scaler = StandardScaler(inputCol='features', outputCol='scaled_features', withStd=True, withMean=True)

# Compute summary statistics by fitting the StandardScaler
scalerModel = scaler.fit(df)

# Normalize each feature to have unit standard deviation.
df = scalerModel.transform(df)

# Build PCA Model
pca = PCAml(k=2, inputCol="scaled_features", outputCol="pca")
model = pca.fit(df)
model_df = model.transform(df)
#
# rows = model.pc.toArray().tolist()
# output_df = spark.createDataFrame(rows)

print(f'{model.explainedVariance}')

