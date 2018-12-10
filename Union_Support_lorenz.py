#! /usr/bin/python3

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from collections import Counter
import spacy
nlp = spacy.load('en_core_web_sm')

class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """
    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        return data_dict[self.key]


class TextStats(BaseEstimator, TransformerMixin):
    """Extract features from each document for DictVectorizer"""

    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        return [{'length': len(text),
                 'num_sentences': text.count('.')}
                for text in posts]


class SubjectBodyExtractor(BaseEstimator, TransformerMixin):
    """Extract the subject & body from a usenet post in a single pass.

    Takes a sequence of strings and produces a dict of sequences.  Keys are
    `subject` and `body`.
    """
    def fit(self, x, y=None):
        return self

    def transform(self, posts):
        # features = np.recarray(shape=(len(posts),),
                               # dtype=[('left', object), ('middle', object), ('right', object),
                                # ('complete', object)])
        features = np.recarray(shape=(len(posts),),
                               dtype=[('left', object), ('middle', object), ('right', object),
                                ('complete', object), ('test', object)])
        for i, text in enumerate(posts):
            left = []
            right = []
            middle = []
            context = []
            for s in text.snippet:
                pos_dict = Counter()
                left.append(s.left)
                right.append(s.right)
                middle.append(s.middle)
                context.append(' '.join((s.left.lower(), s.middle.lower(), s.right.lower())))
                # doc = nlp(s.middle)
                # pos_tags = " ".join([token.pos_ for token in doc])
                # for token in doc:
                    # pos_dict[token.pos_] += 1
                # ratio_prop_noun = (pos_dict["PROPN"] + 1) / (pos_dict["NOUN"] + 1)
                # print(ratio_prop_noun)
            features['left'][i] = " ".join(left)
            features['middle'][i] = " ".join(middle)
            features['right'][i] = " ".join(right)
            features['complete'][i] = " ".join(context)
            # features['test'][i] = pos_tags
            

        return features
