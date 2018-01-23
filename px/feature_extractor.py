# -*- coding: utf-8 -*-

from __future__ import unicode_literals
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse as sp
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


def token_processor(tokens):
    stemmer = SnowballStemmer('english')
    for token in tokens:
        yield stemmer.stem(token)


class FeatureExtractor(TfidfVectorizer):
    """Convert a collection of raw docs to a matrix of TF-IDF features. """

    def __init__(self):
        # see ``TfidfVectorizer`` documentation for other feature
        # extraction parameters.
        super(FeatureExtractor, self).__init__(
                analyzer='char', ngram_range=(1, 3), decode_error='ignore',
                stop_words='english', strip_accents='unicode', max_df=0.7, min_df=0.03)

    def fit(self, X_df, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        X_df : pandas.DataFrame
            a DataFrame, where the text data is stored in the ``statement``
            column.
        """

        super(FeatureExtractor, self).fit(X_df.statement)
        return self

    def fit_transform(self, X_df, y=None):
        return self.fit(X_df).transform(X_df)

    def transform(self, X_df):
        X = super(FeatureExtractor, self).transform(X_df.statement)
        return X

    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))