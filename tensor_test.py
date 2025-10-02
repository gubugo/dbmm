import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import minmax_scale

def find_n_closest_points(points, n):
    n = max(1, min(n, len(points) - 1)) 

    nbrs = NearestNeighbors(n_neighbors=n + 1, algorithm='auto').fit(points)

    _, indices = nbrs.kneighbors(points)

    return indices[:, 1:]

def expand_projection(X, y, *args):
  X = np.load(os.path.join("../dbm-project/data", "mnist", "X.npy"))
  y = np.load(os.path.join("../dbm-project/data", "mnist", "y.npy"))

  X, _, y, _ = train_test_split(
      X, y, train_size=20000, test_size=500, random_state=420, stratify=y
  )

  pca = PCA(n_components=2)
  X = pca.fit_transform(X)
  X = minmax_scale(X, feature_range=(-1,1))

  new_X = np.random.uniform(low=-1, high=1, size=(np.shape(X)[0]//2,np.shape(X)[1]))

  new_y = np.zeros(shape=np.shape(X)[0]//2)-1

### DEBUG ###
  for j in range(10):
    count = 0
    for i in y:
      if i == j:
        count +=1

    print(f"{j}:{count}")
  print(np.shape(X))
  print(np.shape(y))
  print("mm")
#############

  X = np.concatenate((X,new_X))
  y = np.concatenate((y,new_y))
  iter = 0
  while np.any(y == -1) and iter < 100:
    iter += 1
    # sub_v = 0
    # old_X = new_X
    # print(f"here {iter}")
    # print(np.shape(new_X))
    idx_nearest = find_n_closest_points(X,iter+1)
    # print(np.shape(idx_nearest))
    for i,(xv, yv, idx) in enumerate(zip(X, y, idx_nearest)):
      if yv >= 0:
        continue
      for index in idx:
        if y[index] >= 0:
          y[i] = y[index]
          y[index] = -1
          #function that changes the attached values indexes too
          break

  for i,yv in reversed(list(enumerate(y))):
    if yv < 0:
      X = np.delete(X,i, axis=0)
      y = np.delete(y,i, axis=0)
      #function that deletes the extra dims from the attached values


### DEBUG ###
  for j in range(10):
    count = 0
    for i in y:
      if i == j:
        count +=1

    print(f"{j}:{count}")
  print(np.shape(X))
  print(np.shape(y))
  print("mm")
#############

  fig, axs = plt.subplots(1, 1, figsize=(6, 6))
  axs.scatter(X[:,0], X[:,1], c=y, cmap="tab10", s=10)
  plt.show()

  return X, y