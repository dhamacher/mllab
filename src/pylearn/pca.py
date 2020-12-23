from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from time import time
import pandas as pd
import scipy.sparse

df = pd.read_csv(r'..\..\data\USArrests.csv')

label_column = df[df.columns[0]]
feature_columns = df.columns[1:]
n_components = 1

X = df[feature_columns].values

# Convert to sparse matrix
# X = scipy.sparse.csr_matrix(df[feature_columns].values)
print("Performing dimensionality reduction using PCA")
t0 = time()

# https://towardsdatascience.com/working-with-sparse-data-sets-in-pandas-and-sklearn-d26c1cfbe067
pca = PCA(n_components)

# This scaler can also be applied to sparse CSR or CSC matrices by passing with_mean=False to avoid breaking the
# sparsity structure of the data.
scaler = StandardScaler(with_mean=False)
lsa = make_pipeline(scaler,pca)

X = lsa.fit_transform(X)

print("done in %fs" % (time() - t0))
print(f'PCA Components: {pca.components_}')
print(f'PCA Explained Variance: {pca.explained_variance_}')
print(f'PCA Explained Variance Ratio: {pca.explained_variance_ratio_}')
print()

# #############################################################################
# Do the actual clustering
km = KMeans(n_clusters=2, init='k-means++', max_iter=100, n_init=1, verbose=True)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)

print("Homogeneity: %0.3f" % metrics.homogeneity_score(label_column, km.labels_))
print("Completeness: %0.3f" % metrics.completeness_score(label_column, km.labels_))
print("V-measure: %0.3f" % metrics.v_measure_score(label_column, km.labels_))
print("Adjusted Rand-Index: %.3f"
      % metrics.adjusted_rand_score(label_column, km.labels_))
print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, km.labels_, sample_size=1000))

print()