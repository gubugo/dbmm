from sklearn.ensemble import RandomForestClassifier

def make_and_fit_rf(X, y) -> RandomForestClassifier:
    return RandomForestClassifier(
        max_depth=20, 
        random_state=420
    ).fit(X, y)