#! /usr/bin/python3
# Usage:
# python relextract.py -a "train.json.txt" -b "test-covered.json.txt" -p "predicted.txt"

import gzip
import numpy as np
import random
import os
import json
import argparse

from collections import Counter, defaultdict, namedtuple
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, fbeta_score, make_scorer
from sklearn.pipeline import Pipeline, make_pipeline, FeatureUnion
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import FunctionTransformer,LabelEncoder
import numpy as np

from Union_Support import ItemSelector, TextStats, SubjectBodyExtractor


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
    
    
def SelectContext(data, verbose=True):
    only_context_data = []
    for instance in data:
        instance_context = []
        for s in instance.snippet:
            #~ context = ' '.join((s.left, s.middle, s.right))
            instance_context.append(s.left.lower())
            instance_context.append(s.middle.lower())
            instance_context.append(s.right.lower())

        only_context_data.append(' '.join(instance_context))
    if verbose:
        print(len(data))
        print(len(only_context_data))
        print(data[0])
        print(only_context_data[0])
    return only_context_data

    
    
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
    for rel in label_encoder.classes_:
            results[rel] = []
    if verbose:
        print_statistics_header()
    kfold = StratifiedKFold(n_splits = 5, shuffle=True, random_state=0) 
    for train_index, test_index in kfold.split(X, y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
        classifier.fit(X_train, y_train)
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

def predict(clf, test_file, train_data_featurized, train_labels_featurized, le, output_file):
    # Fit final model on the full train data
    clf.fit(train_data_featurized, train_labels_featurized)

    # Predict on test set
    test_data, test_labels = load_data(test_file, verbose=False)
    #~ test_data_featurized = SelectContext(test_data, verbose=False)
    test_label_predicted = clf.predict(test_data)
    # Deprecation warning explained: https://stackoverflow.com/questions/49545947/sklearn-deprecationwarning-truth-value-of-an-array
    test_label_predicted_decoded = le.inverse_transform(test_label_predicted)
    print(test_label_predicted_decoded[:2])
    f = open(output_file, "w+", encoding="utf-8")
    for label in test_label_predicted_decoded:
        f.write(label+'\n')
        
        
# Feature analisys - print N most informative
# !! Make changes in this function when you change the pipleine!!
def printNMostInformative(classifier,label_encoder,N):
    """Prints features with the highest coefficient values, per class"""
    #~ feature_names = classifier.named_steps['dictvectorizer'].get_feature_names()
    feature_names = classifier.named_steps['union'].transformer_list[0][1].named_steps['countvectorizer'].get_feature_names()

    coef = classifier.named_steps['lr'].coef_    
    print(coef.shape)
    for rel in label_encoder.classes_:
        rel_id = label_encoder.transform([rel])[0]
        coef_rel = coef[rel_id]
        coefs_with_fns = sorted(zip(coef_rel, feature_names))
        top_features = coefs_with_fns[-N:]
        print("\nClass {} best: ".format(rel))
        for feat in top_features:
            print(feat)  
            
            
def main():
    parser = argparse.ArgumentParser(description="Relation classification")
    parser.add_argument("-a", dest="train_set", help="Training set for the classification")
    parser.add_argument("-b", dest="test_set", help="Testing set for the classification")
    parser.add_argument("-p", dest="prediction_file", help="Name of the file to write predictions to")
    args = parser.parse_args()
	
    verbose = False
    analysis = False
    evaluate = True
    
    train_file = args.train_set # 'train.json.txt'
    test_file = args.test_set # 'test-covered.json.txt'
    predictions_output_file = args.prediction_file
    
    train_data, train_labels = load_data(train_file, verbose)
    if verbose:
        print('Train set statistics:')
        print_stats(train_labels)
    check_train_data(train_data, train_labels, verbose)
    # Transform dataset to features
    #~ train_data_featurized = ExtractSimpleFeatures(train_data, verbose)
    #~ train_data_featurized = SelectContext(train_data)
    # Transform labels to nimeric values
    le = LabelEncoder()
    train_labels_featurized = le.fit_transform(train_labels)
    # Fit model one vs rest logistic regression
    #~ clf = make_pipeline(CountVectorizer(), LogisticRegression())
    clf = Pipeline([
    
        ('subjectbody', SubjectBodyExtractor()),
    
        ('union', FeatureUnion(
            transformer_list=[
                ('mid_bow', Pipeline([
                    ('selector', ItemSelector(key='middle')),
                    ('countvectorizer', CountVectorizer()),
                ])),
                
                ('left_bow', Pipeline([
                    ('selector', ItemSelector(key='left')),
                    ('countvectorizer', CountVectorizer()),
                ])),
                
                ('right_bow', Pipeline([
                    ('selector', ItemSelector(key='right')),
                    ('countvectorizer', CountVectorizer()),
                ])),
                # ('pos', Pipeline([
                    # ('selector', ItemSelector(key='test')),
                    # ('countvectorizer', CountVectorizer()),
                # ])),
                
                # ('bow', Pipeline([
                    # ('selector', ItemSelector(key='complete')),
                    # ('countvectorizer', CountVectorizer()),
                # ])),
            ],
            
            transformer_weights={
                'mid_bow': 3.0,
                'left_bow': 1.0,
                'right_bow': 1.0,
                # 'pos': 1.0,
                # 'bow': 1.0,
            },
        )),
        
        ('lr', LogisticRegression(C=0.15)),
    
    ])
    
    evaluateCV(clf, le, train_data, train_labels_featurized, evaluate)
    evaluateCV_check(clf,train_data,train_labels_featurized, verbose)
    # Fit final model on the full train data
    if predictions_output_file:
        predict(clf, test_file, train_data, train_labels_featurized, le, predictions_output_file)
    if analysis:
        print("Top features used to predict: ")
        # show the top features
        printNMostInformative(clf,le,3)
        

if __name__ == "__main__":
    main()



