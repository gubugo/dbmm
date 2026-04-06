from matplotlib import pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import umap

from code.utils.data import Load_data, train_test_split_augmented

dataset_name = "reuters"

tab10_cmap = plt.get_cmap("tab10")

X, y = Load_data("data", dataset_name)

X_train, y_train, noise_train, X_test, y_test, noise_test = train_test_split_augmented(X, y, "noise", train_size=6500, test_size=2000, random_state=420)

# tsne = TSNE(
#     n_jobs=4, 
#     random_state=420, 
#     n_components=2
# )

# points = tsne.fit_transform(X_train)

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
points = reducer.fit_transform(X_train)

fig, ax = plt.subplots()

ax.scatter((points[:, 0]-np.min(points[:, 0]))/(np.max(points[:, 0])-np.min(points[:, 0])), 
            (points[:, 1]-np.min(points[:, 1]))/(np.max(points[:, 1])-np.min(points[:, 1])), 
            c=tab10_cmap(y_train), cmap="tab10", s=12, alpha=1.0)

ax.axis("off") 

fig.savefig(f"{dataset_name}.png", bbox_inches="tight", pad_inches=0.0)