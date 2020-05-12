import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class NBTfidfVectorizer(TfidfVectorizer):
    """Class for generating Naive Bayes features with tf-idf priors.
    Can also be used to generate tf-idf only.
    """

    def __init__(self, NB=True, ngram_range=(1, 2), min_df=50,
                 max_df=0.9, max_features=None, sublinear_tf=True):

        # instantiate TfidfVectorizer parent class
        super().__init__(
            ngram_range=ngram_range, min_df=min_df,
            max_df=max_df, max_features=max_features,
            sublinear_tf=sublinear_tf)

        self._NB = NB       # set to False to only generate tf-idf features
        self._r = None      # NB parameters

    def fit(self, X, y=None):
        """Calculate NB and tf-idf parameters """

        # fit and generate TF-IDF features
        X_tfidf = super().fit_transform(X)

        if self._NB and y is not None:
            # get NB features
            p = (X_tfidf[y == 1].sum(0) + 1) / ((y == 1).sum() + 1)
            q = (X_tfidf[y == 0].sum(0) + 1) / ((y == 0).sum() + 1)
            self._r = np.log(p / q)

    def transform(self, X):
        X_tfidf = super().transform(X)
        return X_tfidf.multiply(self._r) if self._NB else X_tfidf

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
