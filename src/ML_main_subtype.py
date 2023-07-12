
# basic modules
import os
import glob 
import random
import copy
from tqdm import tqdm 
from typing import Dict, Tuple
import six 
import argparse

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
from sklearn.ensemble import BaggingClassifier 
from imblearn.over_sampling import * 
from imblearn.under_sampling import * 


# for hyper parameter tuning
import optuna
from optuna.samplers import TPESampler

# for visulaize 
from sklearn.metrics import RocCurveDisplay, DetCurveDisplay
import matplotlib.pyplot as plt


# custom functions 
from utils.data_preprocessing_subtype import *
from utils.ML_utils_subtype import * 
from utils.utils import * 







def argument_setting(): 
    parser = argparse.ArgumentParser() 
    
    parser.add_argument("--meta_data_Dir", default="/scratch/connectome/dhkdgmlghks/kidney_radiomics/meta_data.csv", type=str, required=False, help="Directory of meta data")
    parser.add_argument("--save_Dir", default='/scratch/connectome/dhkdgmlghks/kidney_radiomics/pipeline/result', type=str, required=False, help="Directory to save the result from radiomics feature extraction")
    parser.add_argument("--CT_phase", default=['A', 'N', 'D', 'P'], type=int,required=False, nargs="*", help="CT phases to use", choices=["A", "D", "N", "P"])
    parser.add_argument("--seed", default=1234, type=int, help="seed number")
    parser.add_argument("--train_ratio", default=0.8, type=float, help="")
    parser.add_argument("--sampling_method", default='oversampling', type=str,help="sampling methods for data imbalance")
    parser.add_argument("--model_name", default='XGBoost', type=str, help="Machine Learning Algorithms", choices=['XGBoost',"RandomForest", "linearSVM", "rbfSVM"])
    parser.add_argument("--scaling_method", default='normalization', type=str, help="", choices=['normalization', 'standardization'])
    parser.add_argument("--scoring", default='accuracy', type=str, help="", choices=['accuracy', 'balanced_accuracy','roc_auc', 'roc_auc_ovo'])
    parser.add_argument("--num_cv", default=10, type=int, help="")
    parser.add_argument("--ratio_of_num_PC", default=0.2, type=float, help="")
    parser.add_argument("--best_model_metric", default='ACC', type=str, help="", choices=["ACC", "AUC"])
    parser.add_argument("--optuna_trials", default=200, type=int, help="")
    parser.add_argument("--k_best", default=10, type=int, help="")
    parser.add_argument("--exp_name", default='experiment', type=str, help="")
    
    args = parser.parse_args()
    return args  




if __name__ == '__main__':
    #### setting arguments
    args = argument_setting()


    #### scaler and fold splitter
    # select scaler
    scaler = selecting_scaler(scaling_method=args.scaling_method)
    # select a method for K fold cv
    cv = StratifiedKFold(args.num_cv, shuffle=True, random_state=args.seed)


    #### preparing dataset
    meta_data = pd.read_csv(args.meta_data_Dir)
    dataset = loading_datasets(save_Dir=args.save_Dir, CT_phase=args.CT_phase)
    train_subject_list, test_subject_list = assign_train_test(meta_data=meta_data, train_ratio=args.train_ratio)
    train_X, train_y, test_X, test_y  = preparing_dataset(meta_data=meta_data, dataset=dataset, train_subject_list=train_subject_list, test_subject_list=test_subject_list)
    case_control_count(train_y, test_y)


    #### Feature Selection 1 
    X = np.concatenate([train_X, test_X], axis=0) 
    y = np.concatenate([train_y, test_y], axis=0)
    # rescaling for initial feature selection 
    scaler = selecting_scaler(scaling_method=args.scaling_method)
    X_rescaled = scaler.fit_transform(X)
    selector = SelectFdr(f_classif)
    selector.fit(X_rescaled, y)
    train_X_selected, test_X_selected = selector.transform(train_X), selector.transform(test_X)


    #### PCA whitening 
    pca = PCA(n_components=int(args.ratio_of_num_PC * (train_X_selected.shape[1])), whiten=True)
    pca.fit(train_X_selected)
    train_X_rescaled_selected1 = pca.transform(train_X_selected)
    test_X_rescaled_selected1 = pca.transform(test_X_selected)


    #### Feature Selection 2 and Model Selection with optuna 
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: hyperparameter_tuning_selection(trial, train_X=train_X_rescaled_selected1, train_y=train_y.squeeze(-1).astype(np.int64), model_name=args.model_name, sampling_method=args.sampling_method,scoring=args.scoring,cv=cv, seed=args.seed), n_trials=args.optuna_trials)
    #print('Best trial: score {},\nparams{}'.format(study.best_trial.value, study.best_trial.params)) 


    #### Prediction with the best model 
    best_params = study.best_trial.params

    # make final prediction on the test dataset 
    if args.model_name == 'XGBoost': 
        best_model_optuna = XGBClassifier(**best_params, random_state=args.seed, learning_rate=0.01, n_jobs=-1)
    elif args.model_name == 'RandomForest': 
        best_model_optuna = RandomForestClassifier(**best_params, random_state=args.seed, class_weight='balanced', n_jobs=-1)
    elif args.model_name == 'linearSVM': 
        best_model_optuna = svm.SVC(kernel='linear',C=best_params['C'], gamma=best_params['gamma'], probability=True,  random_state=args.seed, class_weight='balanced')
        best_model_optuna = BaggingClassifier(estimator=best_model_optuna, n_estimators=best_params['n_estimators'],max_samples = 1./best_params['n_estimators'])
    elif args.model_name == 'rbfSVM': 
        best_model_optuna = svm.SVC(kernel='rbf',C=best_params['C'], gamma=best_params['gamma'], probability=True,  random_state=args.seed, class_weight='balanced')
        best_model_optuna = BaggingClassifier(estimator=best_model_optuna, n_estimators=best_params['n_estimators'],max_samples = 1./best_params['n_estimators'])
    # SFS feature selection
    sfs = SequentialFeatureSelector(best_model_optuna, n_features_to_select='auto', direction='backward',tol=10)
    train_X_rescaled_selected2 =sfs.fit_transform(train_X_rescaled_selected1, train_y.squeeze(-1).astype(np.int64))

    # cross validation 
    """
    The reason I applied cross validation once again is that make sure not to suffle train/validation set in hyperparameter tuning and train/validation set in test. 
    If cross validation is not applied in this stage, samples used for validation in hyper parameter tuning could be used for trainining to fit the selected model. (In scikit-learn default setting, cross_validate() do not shuffle data)
    """
    result = cross_validate(best_model_optuna, train_X_rescaled_selected2, train_y.squeeze(-1).astype(np.int64), cv=args.num_cv, scoring=args.scoring, return_estimator=True) # Stratified Cross Validation is default
  
    # getting best model based on the result of cross validation 
    summary = {'train_ACC':[], 'test_ACC':[], 'train_balanced_ACC':[],  'val_ACC': None,'test_balanced_ACC':[], 'train_F1':[], 'test_F1':[]}
    # validation score 
    summary['val_ACC'] = result['test_score'].tolist()
    train_proba = []
    pred_proba = [] 
    for result_model in result['estimator']:
        test_X_rescaled_selected2 = sfs.transform(test_X_rescaled_selected1)
        train_ACC, train_balanced_ACC, _, train_F1, _, _, _, _ = getting_result(result_model, train_X_rescaled_selected2, train_y.squeeze(-1).astype(np.int64))
        summary['train_ACC'].append(train_ACC)
        summary['train_balanced_ACC'].append(train_balanced_ACC)
        summary['train_F1'].append(train_F1)
        train_proba.append(result_model.predict_proba(train_X_rescaled_selected2).tolist())
        test_ACC, test_balanced_ACC, _, test_F1, _, _, _, _ = getting_result(result_model, test_X_rescaled_selected2, test_y.squeeze(-1).astype(np.int64))
        summary['test_ACC'].append(test_ACC)
        summary['test_balanced_ACC'].append(test_balanced_ACC)
        summary['test_F1'].append(test_F1)
        pred_proba.append(result_model.predict_proba(test_X_rescaled_selected2).tolist())
        
    # mean all the results from each fold 
    for k in summary.keys(): 
        summary[k] = np.mean(summary[k])
    mean_train_proba = np.mean(train_proba, axis=0)
    mean_pred_proba = np.mean(pred_proba, axis=0)
    # metrics based on predicted probability are caculated with mean probability from all estimator
    train_AUC, train_sensitivity, train_specificity, train_precision, train_npv = getting_prob_based_result(y=train_y.squeeze(-1).astype(np.int64), proba=mean_train_proba)
    summary['train_AUC'] = train_AUC
    summary['train_sensitivity'] = train_sensitivity
    summary['train_specificity'] = train_specificity
    summary['train_precision'] = train_precision
    summary['train_npv'] = train_npv
    test_AUC, test_sensitivity, test_specificity, test_precision, test_npv = getting_prob_based_result(y=test_y.squeeze(-1).astype(np.int64), proba=mean_pred_proba)
    summary['test_AUC'] = test_AUC
    summary['test_sensitivity'] = test_sensitivity
    summary['test_specificity'] = test_specificity
    summary['test_precision'] = test_precision
    summary['test_npv'] = test_npv
    # save data needed to draw plot 
    summary['train_proba'] = train_proba
    summary['mean_train_proba'] = mean_train_proba.tolist()
    summary['train_y'] = train_y.tolist()
    summary['pred_proba'] = pred_proba
    summary['mean_pred_proba'] = mean_pred_proba.tolist()
    summary['test_y'] = test_y.tolist()
     # save the best model and values 
    save_result(summary=summary, model=best_params, save_Dir=args.save_Dir, args=args)


   

