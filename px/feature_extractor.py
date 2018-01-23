# -*- coding: utf-8 -*-

from __future__ import unicode_literals

from sklearn.feature_extraction.text import TfidfVectorizer
import string
import unicodedata
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.decomposition import LatentDirichletAllocation

# def document_preprocessor(doc):
#     # TODO: is there a way to avoid these encode/decode calls?
#     try:
#         doc = unicode(doc, 'utf-8')
#     except NameError:  # unicode is a default on python 3
#         pass
#     doc = unicodedata.normalize('NFD', doc)
#     doc = doc.encode('ascii', 'ignore')
#     doc = doc.decode("utf-8")
#     return str(doc)

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
                analyzer='word',ngram_range = (1,2),
                stop_words='english', decode_error='replace',
                strip_accents='unicode',
                max_df = 1.0, min_df = 0.0001)
        self.lda = LatentDirichletAllocation(n_components=7, max_iter=20,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=None)

    def fit(self, X_df, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        X_df : pandas.DataFrame
            a DataFrame, where the text data is stored in the ``statement``
            column.
        """
        X_tf = super(FeatureExtractor, self).fit_transform(X_df.statement)
        self.lda.fit(X_tf)
        return self

    def fit_transform(self, X_df, y=None):
        
        super(FeatureExtractor, self).fit(X_df)
        
        X_tf = self.transform(X_df)
        return self.lda.transform(X_tf)

    def transform(self, X_df):
        X = super(FeatureExtractor, self).transform(X_df.statement)
        return self.lda.transform(X)
    
    def build_tokenizer(self):
        """
        Internal function, needed to plug-in the token processor, cf.
        http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes
        """
        tokenize = super(FeatureExtractor, self).build_tokenizer()
        return lambda doc: list(token_processor(tokenize(doc)))

