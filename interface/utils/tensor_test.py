import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler, minmax_scale

def find_n_closest_points(points, n):
    n = max(1, min(n, len(points) - 1)) 

    nbrs = NearestNeighbors(n_neighbors=n + 1, algorithm='auto').fit(points)

    _, indices = nbrs.kneighbors(points)

    return indices[:, 1:]

def expand_projection(X_data, y_class):
  # X = np.load(os.path.join("../dbm-project/data", "mnist", "X.npy"))
  # y = np.load(os.path.join("../dbm-project/data", "mnist", "y.npy"))

  # X_data, _, y_class, _ = train_test_split(
  #     X, y, train_size=20000, test_size=500, random_state=420, stratify=y
  # )

  pca = PCA(n_components=2)
  # print(np.shape(X_data))
  X = pca.fit_transform(X_data)
  X = minmax_scale(X, feature_range=(0,1))

  new_X = np.random.uniform(low=-1, high=1, size=(np.shape(X)[0]//2,np.shape(X)[1]))

  new_y = np.zeros(shape=np.shape(X)[0]//2)-1

  new_X_data = np.zeros(shape=(np.shape(X_data)[0]//2,np.shape(X_data)[1]))

### DEBUG ###
  for j in range(10):
    count = 0
    for i in y_class:
      if i == j:
        count +=1

    print(f"{j}:{count}")
  print(np.shape(X))
  print(np.shape(y_class))
  print("mm")
#############

  X = np.concatenate((X,new_X))
  X_data = np.concatenate((X_data,new_X_data))
  y = np.concatenate((y_class,new_y))
  iter = 0
  while np.any(y == -1) and iter < 50:
    iter += 1
    # sub_v = 0
    # old_X = new_X
    print(f"here {iter}")
    # print(np.shape(new_X))
    idx_nearest = find_n_closest_points(X,np.min((iter+1,10)))
    # print(np.shape(idx_nearest))
    for i,(xv, yv, idx) in enumerate(zip(X, y, idx_nearest)):
      if yv >= 0:
        continue
      for index in idx:
        if y[index] >= 0:
          y[i] = y[index]
          y[index] = -1
          aux = X_data[i]
          X_data[i] = X_data[index]
          X_data[index] = aux
          #function that changes the attached values indexes too
          break

  for i,yv in reversed(list(enumerate(y))):
    print(i)
    if yv < 0:
      X = np.delete(X,i, axis=0)
      y = np.delete(y,i, axis=0)
      X_data = np.delete(X_data,i, axis=0)
      #function that deletes the extra dims from the attached values

  # y = y_class

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

  # fig, axs = plt.subplots(1, 1, figsize=(6, 6))
  # axs.scatter(X[:,0], X[:,1], c=y, cmap="tab10", s=10)
  # plt.show()

  return X_data, X, np.array(y, dtype=np.uint8)

def normalize_data(X):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(X)

def repel_particles_all1(X, y, k=0.01, alpha=0.001, beta=0.01, 
                        n_iter=20, box_limit=1.0, tol=0.05, save_path="frames"):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    N = X.shape[0]
    for it in range(n_iter):
        forces = np.zeros_like(X)

        # 1. couloumb force
        for i in range(N):
            diff = X[i] - X
            dist_sq = np.sum(diff**2, axis=1) + 1e-6
            inv_dist_sq = 1.0 / dist_sq
            force = np.sum(k * diff * inv_dist_sq[:, None], axis=0)
            forces[i] = force

        # 2. radial external pressure
        forces -= alpha * X

        # 3. wall pressure
        wall_force = np.zeros_like(X)
        eps = 1e-3

        # eixo x
        dist_right = box_limit - X[:,0] + eps
        dist_left  = X[:,0] + box_limit + eps
        wall_force[:,0] += -1.0 / dist_right
        wall_force[:,0] +=  1.0 / dist_left 

        # eixo y
        dist_top    = box_limit - X[:,1] + eps
        dist_bottom = X[:,1] + box_limit + eps
        wall_force[:,1] += -1.0 / dist_top
        wall_force[:,1] +=  1.0 / dist_bottom

        forces += beta * wall_force

        # update positions
        X = X + forces
        X = np.clip(X, -box_limit, box_limit)

        # save image
        plt.figure(figsize=(6,6))
        plt.scatter(X[:,0], X[:,1], c=y, cmap="tab10", s=10)
        plt.xlim(-box_limit, box_limit)
        plt.ylim(-box_limit, box_limit)
        plt.title(f"It {it}")
        plt.savefig(f"{save_path}/frame_{it:03d}.png")
        plt.close()

        # stopping condition
        hits = np.sum((np.abs(X) >= box_limit).any(axis=1))
        if hits / N >= tol:
            print(f"Parando na iteração {it}, {hits/N:.1%} dos pontos atingiram o limite.")
            break

    print(f"Parando na iteração {it}, limite de iterações atingido")

    return X


def repel_particles_all2(X, y, k=0.01,
                        n_iter=100, box_limit=1.0, tol=0.05, save_path="frames"):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    N = X.shape[0]
    for it in range(n_iter):
        forces = np.zeros_like(X)

        # 1. couloumb force
        for i in range(N):
            diff = X[i] - X
            dist_sq = np.sum(diff**2, axis=1) + 1e-6
            inv_dist_sq = 1.0 / dist_sq
            force = np.sum(k * diff * inv_dist_sq[:, None], axis=0)
            forces[i] = force

        # update positions
        X = X + forces

        # stopping condition
        #$hits = np.sum((np.abs(X) >= box_limit).any(axis=1))
        #if hits / N >= tol:
        #    print(f"Parando na iteração {it}, {hits/N:.1%} dos pontos atingiram o limite.")
        #    break

    print(f"Parando na iteração {it}, limite de iterações atingido")

    return X

# expand_projection([],[])