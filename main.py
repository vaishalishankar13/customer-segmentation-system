from src.preprocessing import load_data, preprocess_data
from src.clustering import elbow_method, train_kmeans, evaluate_clustering
from src.pca_analysis import perform_pca, get_loadings
from src.visualization import plot_clusters


# Loading Data
df = load_data('data/q2_customers.csv')
print("completed1")

# Preprocessing
scaled_data = preprocess_data(df)
print("completed2")

# Elbow Method
elbow_method(scaled_data)
print("completed3")

# Training Model
clustered_df, centroids = train_kmeans(scaled_data, df, k=3)
print("completed4")

# Evaluatation
evaluate_clustering(scaled_data, clustered_df['cluster'])
print("completed5")

# PCA
pca, pca_data = perform_pca(scaled_data)
print("completed6")

# Loadings
loadings = get_loadings(pca, df.columns[:-1])
print("completed7")

# Visualization
plot_clusters(pca_data, clustered_df['cluster'])
print("completed8")