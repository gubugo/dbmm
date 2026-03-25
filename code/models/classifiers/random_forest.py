from sklearn.ensemble import RandomForestClassifier

def make_and_fit_rf(X, y) -> RandomForestClassifier:
    return RandomForestClassifier(
        max_depth=None, 
        min_samples_leaf=4,
        n_estimators=400,
        n_jobs=3,
        random_state=420,
    ).fit(X, y)