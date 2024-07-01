import pandas as pd
import numpy as np
from functools import partial
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from hyperopt import fmin, tpe, hp, Trials
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import json
from functions import *

def objective(params, X, y):
    '''
    combine k-fold cross validation with hyperparameter optimization
    using hyperopt
    '''

    
    model = XGBRegressor(
        n_estimators = int(params['n_estimators']),
        max_depth = int(params['max_depth']),
        gamma = params['gamma'],
        learning_rate = params['learning_rate'],
        subsample = params['subsample'],
        colsample_bytree = params['colsample_bytree'],
        min_child_weight = params['min_child_weight'],
        eta = params['eta'],
        random_state=42
    )

    pipeline = Pipeline(
    [
        ('encoder', OneHotEncoder(drop=None, handle_unknown='ignore')),
        ('xgboost', model)
    ]
    )
    
    scores = cross_val_score(pipeline, X, y, cv=5, scoring = 'neg_mean_squared_error')
    mse = -np.mean(scores)
    return mse


# Define the hyperparameter search space
space = {
    'n_estimators': hp.quniform('n_estimators', 100, 1000, 1),
    'max_depth': hp.quniform('max_depth', 1, 15, 1),
    'gamma': hp.quniform ('gamma', 0, 1, 0.05),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),
    'subsample': hp.quniform('subsample', 0.5, 1, 0.05),
    'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
    'min_child_weight' : hp.quniform('min_child_weight', 0, 6, 1),
    'eta': hp.quniform('eta', 0.025, 0.5, 0.025)
}



n_bootstraps = 100
max_evals = 50

df = pd.read_csv("../data/BD4QoL_150523_encoded.csv") # original data
bs = pd.read_csv("../data/bootstrap_samples.csv")

# load variable list
variable_list = pd.read_csv("../data/prospective_variables.csv")

# bootstrap variables list
bs_variables = variable_list.copy(deep = True)
bs_variables.loc[len(bs_variables)] = {'variable': 'bs_id', 'type': 'index', 'dtype': 'int64', 'rtype':'int'}

df = preprocess(data = df, features = variable_list, target = "hn4_dv_c30_ghs")
bs = preprocess(data = bs, features = bs_variables, target = "hn4_dv_c30_ghs")

# define the independent variables
#covariates = ["hn1_dv_age_cons", "hn1_na8_cb_sex", "hn1_icd_group_conf", "hn1_tnm_stage_best", "hn1_nb4_cb_comorb_index"]
covariates = variable_list[variable_list['predictor']=='yes'].variable.values.tolist()

# separate predictors and outcome
X_orig = df[covariates]
y_orig = df['hn4_dv_c30_ghs']

# Perform hyperparameter optimization on the original data
trials = Trials()
best = fmin(fn=partial(objective, X=X_orig, y=y_orig), 
            space=space, 
            algo=tpe.suggest, 
            max_evals=max_evals, 
            trials=trials)

best_model = XGBRegressor(
        n_estimators = int(best['n_estimators']),
        max_depth = int(best['max_depth']),
        gamma = best['gamma'],
        learning_rate = best['learning_rate'],
        subsample = best['subsample'],
        colsample_bytree = best['colsample_bytree'],
        min_child_weight = best['min_child_weight'],
        eta = best['eta'],
        random_state=42
    )
    

pipeline_best = Pipeline(
        [
        ('encoder', OneHotEncoder(drop=None, handle_unknown='ignore')),
        ('xgboost', best_model)
        ]
    )
    
pipeline_best.fit(X_orig, y_orig)


# Make predictions on the original data
y_pred_orig = pipeline_best.predict(X_orig)

# Calculate Mean Squared Error
origorig_mse = mean_squared_error(y_orig, y_pred_orig)
origorig_r2 = r2_score(y_orig, y_pred_orig) 

# train in the bootstrap and test in the oob samples
oob_r2 = np.zeros(n_bootstraps)
oob_mse = np.zeros(n_bootstraps)

# train in the bootstrap and test in the bootstrap
bsbs_r2 = np.zeros(n_bootstraps)
bsbs_mse = np.zeros(n_bootstraps)

# train in the bootstrap and test in the original data
bsorig_r2 = np.zeros(n_bootstraps)
bsorig_mse = np.zeros(n_bootstraps)


for bootstrap in range(0, n_bootstraps):
    bs_n = bs[bs["bs_id"] == bootstrap + 1]
    oob = get_oob_samples(df, bs_n.loc[:, "studyid_hn057"])
    
    # bootstrap data
    X_train = bs_n.loc[:, covariates]
    y_train = bs_n.loc[:, "hn4_dv_c30_ghs"]

    # bootstrap out-of-bag samples
    X_oob = oob.loc[:, covariates]
    y_oob = oob.loc[:, "hn4_dv_c30_ghs"]

    # hyperparametrization on the bootstrap
    trials = Trials()
    best = fmin(fn=partial(objective, X=X_train, y=y_train), 
            space=space, 
            algo=tpe.suggest, 
            max_evals=max_evals, 
            trials=trials)
    
    best_model = XGBRegressor(
        n_estimators = int(best['n_estimators']),
        max_depth = int(best['max_depth']),
        gamma = best['gamma'],
        learning_rate = best['learning_rate'],
        subsample = best['subsample'],
        colsample_bytree = best['colsample_bytree'],
        min_child_weight = best['min_child_weight'],
        eta = best['eta'],
        random_state=42
    )
    
    pipeline_best = Pipeline(
        [
        ('encoder', OneHotEncoder(drop=None, handle_unknown='ignore')),
        ('xgboost', best_model)
        ]
    )


    pipeline_best.fit(X_train, y_train)


    # Make predictions on the test data
    predictions_oob = pipeline_best.predict(X_oob)
    predictions_bs = pipeline_best.predict(X_train)
    predictions_orig = pipeline_best.predict(X_orig)

    # Calculate Mean Squared Error
    oob_mse[bootstrap] = mean_squared_error(y_oob, predictions_oob)
    oob_r2[bootstrap] = r2_score(y_oob, predictions_oob) 
    
    bsbs_mse[bootstrap] =  mean_squared_error(y_train, predictions_bs)
    bsbs_r2[bootstrap] = r2_score(y_train, predictions_bs)
    
    bsorig_mse[bootstrap] =  mean_squared_error(y_orig, predictions_orig)
    bsorig_r2[bootstrap] = r2_score(y_orig, predictions_orig)

    

mse_362 = dot632_estimator(origorig_mse, oob_mse)
r2_362 = dot632_estimator(origorig_r2, oob_r2)
    
results = {
           'metrics' : 
            {
               'oob_mse': oob_mse.tolist(),
               'oob_r2': oob_r2.tolist(),
               'origorig_mse': origorig_mse.tolist(), 
               'bsbs_mse': bsbs_mse.tolist(), 
               'bsorig_mse': bsorig_mse.tolist(), 
               'bsorig_r2': bsorig_r2.tolist(),
               'mse_362': mse_362,
               'r2_362': r2_362
            },
            'description': 
            {
                'oob': 'Trained on the BS and evaluated on OOB sample',
                'origorig': 'Trained on the original data and evaluated on the original data',
                'bsbs': 'Trained on the BS and evaluated on the BS sample',
                '_362': 'Bootstrap .362 estimator'
            }
        }



with open('results_extensive.json', 'w') as file:
    json.dump(results, file)