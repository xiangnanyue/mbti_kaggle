# the original codes
# -*- coding: utf-8 -*-
#from __future__ import unicode_literals

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import string
import unicodedata
import re
import nltk
import string
import igraph
import itertools
import unicodedata
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder, MaxAbsScaler
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_str(sentence, stem=True):
    english_stopwords = set(
        [stopword for stopword in stopwords.words('english')])
    punctuation = set(string.punctuation)
    punctuation.update(["``", "`", "..."])
    if stem:
        stemmer = SnowballStemmer('english')
        return list((filter(lambda x: x.lower() not in english_stopwords and
                            x.lower() not in punctuation,
                            [stemmer.stem(t.lower())
                             for t in word_tokenize(sentence)
                             if t.isalpha()])))

    return list((filter(lambda x: x.lower() not in english_stopwords and
                        x.lower() not in punctuation,
                        [t.lower() for t in word_tokenize(sentence)
                         if t.isalpha()])))

def clean_text(text, my_stopwords, punct, remove_stopwords=True, lower_case=True):

    if lower_case:
        text = text.lower()
    text = ''.join(l for l in text if l not in punct)  # remove punctuation (preserving intra-word dashes)
    text = re.sub(' +', ' ', text)  # strip extra white space
    text = text.strip()  # strip leading and trailing white space
    # tokenize (split based on whitespace)
    tokens = text.split(' ')

    if remove_stopwords:
        # remove stopwords
        tokens = [token for token in tokens if token not in my_stopwords]

    return tokens


def pos_filter(tokens):
    # POS tag and retain only nouns and adjectives
    tagged_tokens = pos_tag(tokens)
    tokens_keep = []
    for item in tagged_tokens:
        if (
            item[1] == 'NN' or
            item[1] == 'NNS' or
            item[1] == 'NNP' or
            item[1] == 'NNPS' or
            item[1] == 'JJ' or
            item[1] == 'JJS' or
            item[1] == 'JJR' or
            item[1] == 'WP' or
            item[1] == 'WRB' or
            item[1] == 'WDT' or
            item[1] == 'PRP' or
            item[1] == 'CD' or
            item[1] == 'VBP' or # is are ..
            item[1] == 'VB' or
            item[1] == 'VBZ' or
            item[1] == 'VBD' or
            item[1] == 'VBN' or
            item[1] == 'RB' # verb
        ):
            # keep some kinds of tags
            tokens_keep.append(item[0])

    tokens = tokens_keep

    return tokens, tagged_tokens

def strip_accents_unicode(s):
    s = unicodedata.normalize('NFD', s)
    s = s.encode('ascii', 'ignore')
    s = s.decode("utf-8")
    return str(s)

def stem_words(tokens):
    # apply Porter's stemmer
    stemmer = nltk.stem.PorterStemmer()
    tokens_stemmed = list()
    for token in tokens:
        tokens_stemmed.append(stemmer.stem(token))
    tokens = list(map(lambda x : strip_accents_unicode(x), tokens_stemmed))

    return (tokens)


def terms_to_graph(terms, w):
    '''This function returns a directed, weighted igraph from a list of terms
    (the tokens from the pre-processed text) e.g., ['quick','brown','fox'].
    Edges are weighted based on term co-occurence
    within a sliding window of fixed size 'w'.
    '''
    print(terms)

    if w > len(terms):
        w = len(terms)

    from_to = {}

    # create initial complete graph (first w terms)
    terms_temp = terms[0:w]
    indexes = list(itertools.combinations(range(w), r=2))

    new_edges = []

    for my_tuple in indexes:
        #print(my_tuple, terms_temp)
        new_edges.append(tuple([terms_temp[i] for i in my_tuple]))

    for new_edge in new_edges:
        if new_edge in from_to:
            from_to[new_edge] += 1
        else:
            from_to[new_edge] = 1

    # then iterate over the remaining terms
    for i in range(w, len(terms)):
        considered_term = terms[i]  # term to consider
        terms_temp = terms[(i - w + 1):(i + 1)]  # all terms within sliding window

        # edges to try
        candidate_edges = []
        for p in range(w - 1):
            candidate_edges.append((terms_temp[p], considered_term))

        for try_edge in candidate_edges:
            if try_edge[1] != try_edge[0]:
                # if not self-edge

                # if edge has already been seen, update its weight
                if try_edge in from_to:
                    from_to[try_edge] += 1

                # if edge has never been seen, create it and assign it a unit weight
                else:
                    from_to[try_edge] = 1

    # create empty graph
    g = igraph.Graph(directed=True)

    # add vertices
    sorted_terms = sorted(set(terms))
    g.add_vertices(sorted_terms)
    g.vs["label"] = sorted_terms

    # add edges, direction is preserved since the graph is directed
    g.add_edges(from_to.keys())

    # set edge and vertex weights
    g.es['weight'] = from_to.values()  # based on co-occurence within sliding window
    #g.vs['weight'] = g.strength(weights=from_to.values())  # weighted degree

    return (g)


class FeatureExtractor(TfidfVectorizer):
    """Convert a collection of raw documents to a matrix of TF-IDF features. """

    def __init__(self):
        super(FeatureExtractor, self).__init__(
            input='content', encoding='utf-8',
            decode_error='strict', strip_accents=None, lowercase=True,
            preprocessor=None, tokenizer=None, analyzer='word',
            stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
            ngram_range=(1, 1), max_df=1.0, min_df=1,
            max_features=None, vocabulary=None, binary=False,
            dtype=np.int64, norm='l2', use_idf=True, smooth_idf=True,
            sublinear_tf=False)

    def fit(self, X_df, y=None):
        """Learn a vocabulary dictionary of all tokens in the raw documents.

        Parameters
        ----------
        raw_documents : iterable
            An iterable which yields either str, unicode or file objects.
        Returns
        -------
        self
        """
        # deal with all the statements feature
        self._feat = np.array([' '.join(
            clean_str(strip_accents_unicode(dd))) for dd in X_df.posts])

        super(FeatureExtractor, self).fit(self._feat)

        return self

    def fit_transform(self, X_df, y=None):
        self.fit(X_df)

        return self.transform(self.X_df)

    def transform(self, X_df):
        X = np.array([' '.join(clean_str(strip_accents_unicode(dd)))
                      for dd in X_df.statement])
        #X = X_df[['source', 'researched_by']]
        check_is_fitted(self, '_feat', 'The tfidf vector is not fitted')
        X = super(FeatureExtractor, self).transform(X)

        return X.todense()


