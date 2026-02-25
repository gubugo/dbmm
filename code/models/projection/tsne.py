
import os
from joblib import dump, load
from sklearn.manifold import TSNE
from sklearn.preprocessing import minmax_scale


def load_or_project_tsne(X_train, X_test, dir, method):
    tsne = TSNE(
        n_jobs=4, 
        random_state=420, 
        n_components=(2 if method=="noise" else 4)
    )

    if os.path.exists(f'{dir}/tsneData2d_train.joblib'):
        proj_train = load(f'{dir}/tsneData2d_train.joblib')
    else:
        proj_train = tsne.fit_transform(X_train)
        proj_train = minmax_scale(proj_train, feature_range=(-1,1))
        dump(proj_train, f'{dir}/tsneData2d_train.joblib')
    
    if os.path.exists(f'{dir}/tsneData2d_test.joblib'):
        X_model_res = load(f'{dir}/tsneData2d_test.joblib')
    else:
        # X_model_res = tsne_proj
        proj_test = tsne.fit_transform(X_test)
        proj_test = minmax_scale(proj_test, feature_range=(-1,1))
        dump(proj_test, f'{dir}/tsneData2d_test.joblib')

    return proj_train, proj_test