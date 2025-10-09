
from sklearn.decomposition import PCA
from sklearn.preprocessing import minmax_scale

def get_augmentation_pca(X):
    pca = PCA(n_components=2)
    augmentation = pca.fit_transform(X)
    return minmax_scale(augmentation, feature_range=(-1,1))