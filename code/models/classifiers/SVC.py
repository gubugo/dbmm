from sklearn.svm import LinearSVC

def make_and_fit_svc(X, y) -> LinearSVC:
    return LinearSVC(
        tol=1e-4,
        random_state=420,
    ).fit(X, y)