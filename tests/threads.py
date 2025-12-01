from matplotlib import pyplot as plt
import numpy as np

side = 2

# Suppose your matrices are A1, A2, ..., A9
di_list = np.zeros((side**2,100,100))

for x_c in range(side):
    for y_c in range(side):
        a = np.load(f'center_15x/dbm_4d_(101)_120_5100_{x_c}_{y_c}.npy')
        di_list[x_c*2+y_c] = a.reshape((100,100))

big = np.block([
    di_list[0:2],
    di_list[2:4],
    # di_list[10:15],
    # di_list[15:20],
    # di_list[20:25],
])

# big = np.block(blocks)

fig_id, ax_id = plt.subplots(1,1,figsize=(50,50))

cmap = plt.get_cmap('jet', int(5-2+1))
            
ax_id.imshow(
    big.reshape((side*100, side*100,1)),
    cmap=cmap,
    interpolation="none",
    resample=False,
    vmin=2,
    vmax=5,
)

ax_id.axis("off")  

fig_id.savefig(f"fth.png", bbox_inches="tight", pad_inches=0.0)