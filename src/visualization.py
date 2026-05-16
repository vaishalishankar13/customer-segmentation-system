import matplotlib.pyplot as plt


def plot_clusters(pca_data, clusters):
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(
        pca_data[:,0],
        pca_data[:,1],
        c=clusters,
        cmap='viridis'
    )
    plt.title('Customer Segmentation using PCA')
    plt.xlabel('PC1')
    plt.ylabel('PC2')

    plt.legend(*scatter.legend_elements(), title='Clusters')

    plt.savefig('outputs/pca_plot.png')
    plt.show()