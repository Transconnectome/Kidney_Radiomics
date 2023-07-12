# machine learning algorithms 
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import BaggingClassifier 

import optuna 

def get_XGBoost(trial, seed): 
    optuna_param = {
        "n_estimators": trial.suggest_int('n_estimators', 1, 200),
        "max_depth": trial.suggest_int('max_depth', 3, 18),
        'gamma': trial.suggest_float('gamma', 0.01, 20),
        'learning_rate': 0.01,
        'colsample_bytree': trial.suggest_discrete_uniform('colsample_bytree', 0.5, 1, 0.1),
        'lambda': trial.suggest_loguniform('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_loguniform('alpha', 1e-3, 10.0),
        'subsample': trial.suggest_categorical('subsample', [ 0.7, 0.8, 0.9]),
        'random_state': seed,
        'n_jobs': -1
    }  
    model = XGBClassifier(**optuna_param)
    return  model


def get_RandomForest(trial, seed):
    optuna_param = {
        "n_estimators": trial.suggest_int('n_estimators', 1, 200),
        "max_depth": trial.suggest_int('max_depth', 3, 16),
        "class_weight": "balanced",
        'max_samples': trial.suggest_categorical('max_samples', [0.6, 0.7, 0.8, 1.0]),
        'random_state': seed,
        'n_jobs': -1
    }  
    model = RandomForestClassifier(**optuna_param)
    return model


def get_linearSVM(trial, seed): 
    optuna_param = {
        "kernel": 'linear',
        "C": trial.suggest_float('C', 0.1, 100),
        'gamma': trial.suggest_float('gamma', 0.01, 20),
        "class_weight": "balanced",
        "probability": True,
        'random_state': seed
    }  
    
    bagging_param = {
        "n_estimators": trial.suggest_int('n_estimators', 1, 10),
        "n_jobs": -1,
    } 
    
    model = svm.SVC(**optuna_param)
    model = BaggingClassifier(estimator=model, max_samples = 1./bagging_param['n_estimators'], **bagging_param)
    return model


def get_rbfSVM(trial, seed): 
    optuna_param = {
        "kernel": 'rbf',
        "C": trial.suggest_float('C', 0.1, 100),
        'gamma': trial.suggest_float('gamma', 0.01, 20),
        "class_weight": "balanced",
        "probability": True,
        'random_state': seed
    }  
    bagging_param = {
        "n_estimators": trial.suggest_int('n_estimators', 1, 10),
        "n_jobs": -1,
    } 
    model = svm.SVC(**optuna_param)
    model = BaggingClassifier(estimator=model, max_samples = 1./bagging_param['n_estimators'], **bagging_param)
    return model