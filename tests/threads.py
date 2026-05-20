from matplotlib import pyplot as plt
import numpy as np

n_pieces = 5
side = 60

# Suppose your matrices are A1, A2, ..., A9
di_list = np.zeros((n_pieces**2,side,side))

print(np.shape(di_list))

for x_c in range(n_pieces):
    for y_c in range(n_pieces):
        a = np.load(f'_ID_res3/dbm_4d_({side+1})_120_{x_c}_{y_c}.npy')
        di_list[x_c*n_pieces+y_c] = a.reshape((side,side))

big = np.block([
    di_list[0:5],
    di_list[5:10],
    di_list[10:15],
    di_list[15:20],
    di_list[20:25],
])

print(np.shape(big))

print(np.mean(big.flatten()))

# big = np.block(blocks)

fig_id, ax_id = plt.subplots(1,1,figsize=(50,50))

cmap = plt.get_cmap('jet', int(5-2+1))
            
ax_id.imshow(
    big.reshape((n_pieces*side, n_pieces*side,1)),
    cmap=cmap,
    interpolation="none",
    resample=False,
    vmin=2,
    vmax=5,
)

ax_id.axis("off")  

fig_id.savefig(f"image_composite_reuters11.png", bbox_inches="tight", pad_inches=0.0)

#ID_X
# HAR     = 53.9364
# reuters = 91.7858
# fmnist  = 58.3520

#ID_X'
# HAR     = 5.4342
# reuters = 6.0792
# fmnist  = 4.9609

#ID_Q

# dbm sampling
# HAR     = 2.00
# reuters = 2.01
# fmnist  = 2.00

# pixel sampling
# HAR     = 2.00
# reuters = 2.01
# fmnist  = 2.00

# composite sampling
# HAR     = 3.4785
# reuters = 3.6555
# fmnist  = 3.8454