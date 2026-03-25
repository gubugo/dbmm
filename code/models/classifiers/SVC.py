from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

def make_and_fit_svc(X, y) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(LinearSVC(
        # C=1.0,        
        # kernel='linear',       # 'linear', 'rbf', 'poly', 'sigmoid'
        # gamma=0.01,         # kernel coefficient
        # max_iter=5000,
        # tol=1e-5,
        random_state=420,
        # probability=True,
        verbose=True,
    )).fit(X, y)