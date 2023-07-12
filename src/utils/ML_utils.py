# basic modules
import os
import glob 
import random
import copy
from tqdm import tqdm 
from typing import Dict, Tuple
import six 

# for data processing
import numpy as np
import pandas as pd

# for radiomics features 
"""
if you running ipython notebook with python version 3.7 and install pyradiomics, you could activate those two lines below. 
However, sklearn version 1.1+, which could be only compatible with python 3.8+, only support "n_features_to_select='auto'" option in sklearn.feature_selection.SequentialFeatureSelector. 
Thus, it is highly recommend to deactivate two below lines and skip Feature Extraction stage and perform Featrue Extraction stage with a python script "feature_extraction.py"
"""
#import SimpleITK as sitk   # 
#from radiomics import featureextractor, getTestCase

# for machine learning experiments
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFECV, SelectFdr, SelectPercentile ,mutual_info_classif, f_classif
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, make_scorer, balanced_accuracy_score
from imblearn.over_sampling import * 
from imblearn.under_sampling import * 

# machine learning algorithms 
from utils.model_config import * 

# for hyper parameter tuning
import optuna
from optuna.samplers import TPESampler

# for visulaize 
from sklearn.metrics import RocCurveDisplay, DetCurveDisplay
import matplotlib.pyplot as plt




def selecting_scaler(scaling_method): 
    if scaling_method == 'normalization': 
        scaler = StandardScaler()
    elif scaling_method == 'standardization':
        scaler = MinMaxScaler()
    return scaler


def case_control_count(train_y, test_y): 
    test_case = test_y.squeeze(-1).tolist().count(1)
    test_control = test_y.squeeze(-1).tolist().count(0)
    train_case = train_y.squeeze(-1).tolist().count(1)
    train_control = train_y.squeeze(-1).tolist().count(0)
    print("Train (case/control): {}/{}. Test (case/control): {}/{}.".format(train_case, train_control, test_case, test_control))


def custom_scorer(y_true, y_pred):
    tn, fp, fn, tp  = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = (tp / (tp + fn))
    specificity = (tn / (tn + fp))
    return specificity





def sampling_and_CrossValidation(train_X, train_y, model, cv, sampling_method, scoring, seed): 
    scores = []
    if sampling_method == 'oversampling': 
        for train, val in cv.split(train_X, train_y):
            X_samp , y_samp = SMOTE(random_state=seed).fit_resample(train_X[train], train_y[train])
            model.fit(X_samp , y_samp)
            if scoring == 'roc_auc':
                pred_proba = model.predict_proba(train_X[val])
                scores.append(roc_auc_score(train_y[val], pred_proba[:,1]))
            elif scoring == 'accuracy': 
                pred = model.predict(train_X[val])
                scores.append(accuracy_score(train_y[val], pred))
            elif scoring == 'specificity': 
                pred = model.predict(train_X[val])
                scores.append(custom_scorer(train_y[val], pred))
        
    elif sampling_method == 'undersampling': 
        for train, val in cv.split(train_X, train_y):
            X_samp , y_samp = ClusterCentroids(random_state=seed).fit_resample(train_X[train], train_y[train])
            model.fit(X_samp , y_samp)
            if scoring == 'roc_auc':
                pred_proba = model.predict_proba(train_X[val])
                scores.append(roc_auc_score(train_y[val], pred_proba[:,1]))
            elif scoring == 'accuracy': 
                pred = model.predict(train_X[val])
                scores.append(accuracy_score(train_y[val], pred))
            elif scoring == 'specificity': 
                pred = model.predict(train_X[val])
                scores.append(custom_scorer(train_y[val], pred))
    return np.array(scores)




def hyperparameter_tuning_selection(trial: optuna.trial.Trial, train_X, train_y, model_name='XGBoost', cv=None, sampling_method=None,scoring='accuracy',seed=1234):
    if model_name == 'XGBoost':
        model = get_XGBoost(trial, seed)
    elif model_name == 'RandomForest': 
        model = get_RandomForest(trial, seed)
    elif model_name == 'linearSVM': 
        model = get_linearSVM(trial, seed)
    elif model_name == 'rbfSVM': 
        model = get_rbfSVM(trial, seed)

    sfs = SequentialFeatureSelector(model, n_features_to_select='auto', direction='backward', scoring=scoring, tol=10, n_jobs=-1)
    train_X_selected2 = sfs.fit_transform(train_X, train_y)
    
    if sampling_method:
        scores = sampling_and_CrossValidation(train_X_selected2, train_y, model, cv, sampling_method, scoring, seed)
        return scores.mean() 
    else:
        if scoring == 'roc_auc':
            scores = cross_val_score(model, train_X_selected2, train_y, cv=cv, scoring='roc_auc')
        elif scoring == 'accuracy':
            scores = cross_val_score(model, train_X_selected2, train_y, cv=cv, scoring='accuracy')
        elif scoring == 'balanced_accuracy': 
            scores = cross_val_score(model, train_X_selected2, train_y, cv=cv, scoring='balanced_accuracy')
        elif scoring == 'specificity': 
            scores = cross_val_score(model, train_X_selected2, train_y, cv=cv, scoring=make_scorer(custom_scorer))   
        return scores.mean()


def selecting_best_model(result: dict): 
    assert 'estimator' in result.keys()
    best_model = result['estimator'][np.argmax(result['test_score'])]
    return best_model


def getting_result(estimator, X, y, threshold=0.8):
    pred = estimator.predict(X)
    pred_proba = estimator.predict_proba(X)
    pred_diff_threshold = (pred_proba[:, 1] >= threshold).astype(int)

    ACC = accuracy_score(y, pred)
    balanced_ACC = balanced_accuracy_score(y, pred)
    AUC = roc_auc_score(y, pred_proba[:, 1])
    F1 = f1_score(y, pred)

    tn, fp, fn, tp  = confusion_matrix(y, pred_diff_threshold).ravel()
    sensitivity = (tp / (tp + fn))
    specificity = (tn / (tn + fp))
    precision  = (tp / (tp + fp))
    npv = (tn / (tn + fn))
    return ACC, balanced_ACC, AUC, F1, sensitivity, specificity, precision, npv


def getting_prob_based_result(y, proba, threshold=0.8):
    pred_diff_threshold = (proba[:, 1] >= threshold).astype(int)
    AUC = roc_auc_score(y, proba[:, 1])
    tn, fp, fn, tp  = confusion_matrix(y, pred_diff_threshold).ravel()
    sensitivity = (tp / (tp + fn))
    specificity = (tn / (tn + fp))
    precision  = (tp / (tp + fp))
    npv = (tn / (tn + fn))
    return  AUC, sensitivity, specificity, precision, npv


def get_best_result(result_summary, best_model_metric): 
    best_model_idx = np.argmax(result_summary[best_model_metric])
    best_result = {}
    for k in result_summary.keys(): 
        best_result[k] = result_summary[k][best_model_idx]
    return best_result 


def get_best_model(result_summary, estimator, best_model_metric): 
    best_model_idx = np.argmax(result_summary[best_model_metric])
    return estimator[best_model_idx] 
