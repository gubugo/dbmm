import time
import numpy as np

start_time = time.perf_counter()
points_per_dim = 17  # number of evenly spaced points in each dimension

# 1D grid
x = np.linspace(-1, 1, points_per_dim)
print(x)

# 4D grid
X = np.stack(np.meshgrid(x, x, x, x, indexing='ij'), axis=-1)

# Reshape into list of 4D points
samples = X.reshape(-1, 4)

print(samples.shape)  # (points_per_dim**4, 4)

end_time = time.perf_counter()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")