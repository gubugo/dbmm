from sklearn.svm import SVC

def make_and_fit_svc(X, y) -> SVC:
    return SVC(
        # C=1.0,              # regularization
        # kernel='rbf',       # 'linear', 'rbf', 'poly', 'sigmoid'
        # gamma=0.01,         # kernel coefficient
        # max_iter=5000,      # equivalent to -m (kind of)
        # tol=1e-5,
        # random_state=420,
        verbose=True,
    ).fit(X, y)