from sklearn.ensemble import RandomForestClassifier

def make_and_fit_rf(X, y) -> RandomForestClassifier:
    return RandomForestClassifier(
        max_depth=30, 
        min_samples_leaf=2,
        n_estimators=200,
        random_state=420,

    ).fit(X, y)