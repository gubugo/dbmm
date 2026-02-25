
def show_single_dbm(dbm, i, j, ax):
    ax[j,i].imshow(
        dbm,
        origin="lower",
        interpolation="none",
        resample=False,
    )
    ax[j,i].axis("off") 
