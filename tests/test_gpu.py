import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.calibration import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


# print(tf.test.is_gpu_available())
# print(tf.test.is_gpu_available(cuda_only=True))

# def Load_data(path, dataset):
#     X = np.load(os.path.join(path, dataset, "X.npy"))
#     y = np.load(os.path.join(path, dataset, "y.npy"))
#     return X, y

# data_dir = "./data/"
# d = "mnist"
# X, y = Load_data(data_dir, d)

# X_train, _, y_train, _ = train_test_split(
#     X, y, train_size=30000, test_size=1250, random_state=420, stratify=y
# )

# _, X_test, _, y_test = train_test_split(
#     X_train, y_train, train_size=10000, test_size=1250, random_state=420, stratify=y_train
# )

# pca = PCA(n_components=2)

# # pca.fit(X_train)

# X_new = pca.fit_transform(X_train)

# print(X_new)




def save_dataset(name, X, y):
    print(name, X.shape)
    base_dir = "./data/"
    lenc = LabelEncoder()
    y = lenc.fit_transform(y)

    for c in np.unique(y):
        print("-->", c, np.count_nonzero(y == c))

    dir_name = os.path.join(base_dir, name)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X.astype("float32"))

    np.save(os.path.join(dir_name, "X.npy"), X)
    np.save(os.path.join(dir_name, "y.npy"), y)

    np.savetxt(os.path.join(dir_name, "X.csv.gz"), X, delimiter=",")
    np.savetxt(os.path.join(dir_name, "y.csv.gz"), y, delimiter=",")


fig, ax = plt.subplots(figsize=(20, 20))

gaussian_noise = tf.random.normal(shape=(1000, 2), mean=0.0, stddev=0.5)
X1 = np.array(gaussian_noise)
y1 = np.array(np.zeros((1000)))

cmap = plt.get_cmap('tab10')

for cl in np.unique(y1):
    ax.scatter(X1[y1 == cl, 0], X1[y1 == cl, 1], c=[cmap(cl)], label=cl, s=375, alpha=0.8)
    # ax.axis("off")

gaussian_noise = tf.random.normal(shape=(1000, 2), mean=1.0, stddev=0.5)
X2 = np.array(gaussian_noise)
y2 = np.array(np.ones((1000)))
for cl in np.unique(y2):
    ax.scatter(X2[y2 == cl, 0], X2[y2 == cl, 1], c=[cmap(cl)], label=cl, s=375, alpha=0.8)
    # ax.axis("off")

gaussian_noise = tf.random.normal(shape=(1000, 2), mean=(-1.0,1.0), stddev=0.5)
X3 = np.array(gaussian_noise)
y3 = np.array(np.ones((1000),dtype=np.uint8)+np.ones((1000),dtype=np.uint8))
for cl in np.unique(y3):
    ax.scatter(X3[y3 == cl, 0], X3[y3 == cl, 1], c=[cmap(cl)], label=cl, s=375, alpha=0.8)
    # ax.axis("off")

fig.savefig("test5")

save_dataset("test", np.concatenate((X1,X2,X3)), np.concatenate((y1,y2,y3)))

plt.close("all")
del fig
del ax