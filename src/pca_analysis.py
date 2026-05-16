from sklearn.decomposition import PCA
import pandas as pd


def perform_pca(scaled_data):
    pca = PCA(n_components=2)

    pca_data = pca.fit_transform(scaled_data)

    explained_variance = pca.explained_variance_ratio_

    print('Explained Variance Ratio:')
    print(explained_variance)

    return pca, pca_data

def get_loadings(pca, columns):
    loadings = pd.DataFrame(
        pca.components_,
        columns=columns,
        index=['PC1', 'PC2'])

    print(loadings)

    return loadings

