from sklearn.naive_bayes import GaussianNB

def make_and_fit_nbg(X, y) -> GaussianNB:
    return GaussianNB().fit(X, y)