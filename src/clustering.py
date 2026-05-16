from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

def elbow_method(scaled_data):
    wcss = []

    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)

    elbow_df = pd.DataFrame({
        'K': range(1, 11),
        'WCSS': wcss
    })

    plt.figure(figsize=(8,5))
    sns.lineplot(data=elbow_df, x='K', y='WCSS', marker='o')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')

    plt.savefig('outputs/elbow_plot.png')
    plt.show()
    
def train_kmeans(scaled_data, original_df, k=3):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)
    original_df['cluster'] = clusters
    #saving the model
    joblib.dump(kmeans, 'models/kmeans_model.pkl')
    centroids = pd.DataFrame(kmeans.cluster_centers_)
    original_df.to_csv('outputs/clustered_customers.csv', index=False)
    return original_df, centroids

from sklearn.metrics import silhouette_score
def evaluate_clustering(scaled_data, clusters):
    score = silhouette_score(scaled_data, clusters)

    print('Silhouette Score:', score)