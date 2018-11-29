#! /usr/bin/python3

import gzip
import numpy as np
import random
import os
import json

from collections import Counter, defaultdict, namedtuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, make_scorer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import FunctionTransformer,LabelEncoder
import numpy as np


############################################################################################
# 1. LOAD DATA
############################################################################################

PairExample = namedtuple('PairExample',
    'entity_1, entity_2, snippet')
Snippet = namedtuple('Snippet',
    'left, mention_1, middle, mention_2, right, direction')
def load_data(file, verbose=True):
    f = open(file,'r', encoding='utf-8')
    data = []
    labels = []
    for i,line in enumerate(f):
        instance = json.loads(line)
        if i==0:
            if verbose:
                print('json example:')
                print(instance)
        #'relation, entity_1, entity_2, snippet' fileds for each example
        #'left, mention_1, middle, mention_2, right, direction' for each snippet
        instance_tuple = PairExample(instance['entity_1'],instance['entity_2'],[])
        for snippet in instance['snippet']:
            try:
                snippet_tuple = Snippet(snippet['left'],snippet['mention_1'],
                                        snippet['middle'], 
                                        snippet['mention_2'],snippet['right'],
                                        snippet['direction'])
                instance_tuple.snippet.append(snippet_tuple)
            except:
                print(instance)
        if i==0:
            if verbose:
                print('\nexample transformed as a named tuple:')
                print(instance_tuple)
        data.append(instance_tuple)
        labels.append(instance['relation'])
    return data,labels
    
    
# Statistics over relations
def print_stats(labels):
    labels_counts = Counter(labels)
    print('{:20s} {:>10s} {:>10s}'.format('', '', 'rel_examples'))
    print('{:20s} {:>10s} {:>10s}'.format('relation', 'examples', '/all_examples'))
    print('{:20s} {:>10s} {:>10s}'.format('--------', '--------', '-------'))
    for k,v in labels_counts.items():
        print('{:20s} {:10d} {:10.2f}'.format(k, v, v /len(labels)))
    print('{:20s} {:>10s} {:>10s}'.format('--------', '--------', '-------'))
    print('{:20s} {:10d} {:10.2f}'.format('Total', len(labels), len(labels) /len(labels)))
    
    
def check_train_data(train_data, train_labels, verbose=True):
    # check that each entity pair is assigned only one relation
    pair_dict={}
    rel_dict={}
    for example, label in zip(train_data,train_labels):
        if (example.entity_1,example.entity_2) not in pair_dict.keys():
            pair_dict[(example.entity_1,example.entity_2)] = [label]
        else:
            pair_dict[(example.entity_1,example.entity_2)].append(label)
            print("Multiple labels!")
            print(example.entity_1,example.entity_2,label)
        if label not in rel_dict.keys():
            rel_dict[label] = [example]
        else:
            rel_dict[label].append(example)
    print("Done checking dictionary")  
        
    # example for each relation
    if verbose:
        for rel in rel_dict.keys():
            ex = rel_dict[rel][0]
            print(rel,ex.entity_1,ex.entity_2)
            
            
###########################################################################################
# 2. EXTRACT FEATURES and BUILD CLASSIFIER
###########################################################################################

# Extract two simple features
def ExractSimpleFeatures(data, verbose=True):
    featurized_data = []
    for instance in data:
        featurized_instance = {'mid_words':'', 'distance':np.inf}
        for s in instance.snippet:
            if len(s.middle.split()) < featurized_instance['distance']:
                featurized_instance['mid_words'] = s.middle
                featurized_instance['distance'] = len(s.middle.split())
        featurized_data.append(featurized_instance)
    if verbose:
        print(len(data))
        print(len(featurized_data))
        print(data[0])
        print(featurized_data[0])
    return featurized_data
    
    
##################################################################################################
# 2. TRAIN CLASSIFIER AND EVALUATE (CV)
##################################################################################################

def print_statistics_header():
    print('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        'relation', 'precision', 'recall', 'f-score', 'support'))
    print('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        '-' * 18, '-' * 9, '-' * 9, '-' * 9, '-' * 9))

def print_statistics_row(rel, result):
    print('{:20s} {:10.3f} {:10.3f} {:10.3f} {:10d}'.format(rel, *result))

def print_statistics_footer(avg_result):
    print('{:20s} {:>10s} {:>10s} {:>10s} {:>10s}'.format(
        '-' * 18, '-' * 9, '-' * 9, '-' * 9, '-' * 9))
    print('{:20s} {:10.3f} {:10.3f} {:10.3f} {:10d}'.format('macro-average', *avg_result))

def macro_average_results(results):
    avg_result = [np.average([r[i] for r in results.values()]) for i in range(3)]
    avg_result.append(np.sum([r[3] for r in results.values()]))
    return avg_result

def average_results(results):
    avg_result = [np.average([r[i] for r in results]) for i in range(3)]
    avg_result.append(np.sum([r[3] for r in results]))
    return avg_result
    
def evaluateCV(classifier, label_encoder, X, y, verbose=True):
    results = {}
    for rel in le.classes_:
            results[rel] = []
    if verbose:
        print_statistics_header()
    kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0) 
    for train_index, test_index in kfold.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
        clf.fit(X_train, y_train)
        pred_labels = classifier.predict(X_test)
        stats = precision_recall_fscore_support(y_test, pred_labels, beta=0.5)
        #print(stats)
        for rel in label_encoder.classes_:
            rel_id = label_encoder.transform([rel])[0]
        #print(rel_id,rel)
            stats_rel = [stat[rel_id] for stat in stats]
            results[rel].append(stats_rel)
    for rel in label_encoder.classes_:
        results[rel] = average_results(results[rel])
        if verbose:
            print_statistics_row(rel, results[rel])
    avg_result = macro_average_results(results)
    if verbose:
        print_statistics_footer(avg_result)
    return avg_result[2]  # return f_0.5 score as summary statistic
    

# A check for the average F1 score
f_scorer = make_scorer(fbeta_score, beta=0.5, average='macro')
def evaluateCV_check(classifier, X, y, verbose=True):
    if verbose:
        kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0) 
        scores = cross_val_score(classifier, X, y, cv=kfold, scoring = f_scorer)
        print("\nCross-validation scores (StratifiedKFold): ", scores)
        print("Mean cv score (StratifiedKFold): ", scores.mean())
        
        
#########################################################################################
# 4. TEST PREDICTIONS and ANALYSIS
#########################################################################################

def predict(clf, test_file):
    # Fit final model on the full train data
    clf.fit(train_data_featurized, train_labels_featurized)

    # Predict on test set
    test_data, test_labels = load_data(test_file, verbose=False)
    test_data_featurized = ExractSimpleFeatures(test_data, verbose=False)
    test_label_predicted = clf.predict(test_data_featurized)
    # Deprecation warning explained: https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
    test_label_predicted_decoded = le.inverse_transform(test_label_predicted)
    print(test_label_predicted_decoded[:2])
    f = open("test_labels.txt", 'w', encoding="utf-8")
    for label in test_label_predicted_decoded:
        f.write(label+'\n')
        
        
# Feature analisys - print N most informative
# !! Make changes in this function when you change the pipleine!!
def printNMostInformative(classifier,label_encoder,N):
    """Prints features with the highest coefficient values, per class"""
    feature_names = classifier.named_steps['dictvectorizer'].get_feature_names()

    coef = classifier.named_steps['logisticregression'].coef_    
    print(coef.shape)
    for rel in label_encoder.classes_:
        rel_id = label_encoder.transform([rel])[0]
        coef_rel = coef[rel_id]
        coefs_with_fns = sorted(zip(coef_rel, feature_names))
        top_features = coefs_with_fns[-N:]
        print("\nClass {} best: ".format(rel))
        for feat in top_features:
            print(feat)        
        

if __name__ == "__main__":
    # can be replace with argparse later
    verbose = False
    analysis = True
    train_file = 'train.json.txt'
    test_file = 'test-covered.json.txt'
    train_data, train_labels = load_data(train_file, verbose)
    if verbose:
        print('Train set statistics:')
        print_stats(train_labels)
    check_train_data(train_data, train_labels, verbose)
    # Transform dataset to features
    train_data_featurized = ExractSimpleFeatures(train_data, verbose)
    # Transform labels to nimeric values
    le = LabelEncoder()
    train_labels_featurized = le.fit_transform(train_labels)
    # Fit model one vs rest logistic regression    
    clf = make_pipeline(DictVectorizer(), LogisticRegression())
    evaluateCV(clf, le, train_data_featurized, train_labels_featurized, verbose)
    evaluateCV_check(clf,train_data_featurized,train_labels_featurized, verbose)
    # Fit final model on the full train data
    predict(clf, test_file)
    if analysis:
        print("Top features used to predict: ")
        # show the top features
        printNMostInformative(clf,le,3)



