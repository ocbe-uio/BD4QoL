import pandas as pd
import numpy as np
from functools import partial
from sklearn import metrics
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from hyperopt import fmin, tpe, hp, Trials
import os
import json
from functions import *
import pickle

def neg_roc_auc_score(y_true, y_pred, **kwargs):
    return -metrics.roc_auc_score(y_true, y_score = y_pred, **kwargs)

def objective(params, X, y, n_splits):
    """
    this is the objective function that will be minimized by hyperopt
    to estimate the best hyperparameters, in this case only the l1-penalizaton alpha
    """
    # load preprocessor pipeline
    preprocessor = load_preprocessor(X)

    model = LogisticRegression(
        penalty='l1', 
        solver='liblinear', 
        C = params['alpha'],
        max_iter = 10000
        )

    # define pipeline
    pipeline = Pipeline( steps = [
                    ('preprocessing', preprocessor),
                    ('regressor', model)]
                     )

    losses = cross_validate(X, y, model = pipeline, loss = neg_roc_auc_score, n_splits = n_splits)

    mean_loss = np.mean(losses)
    
    return mean_loss


n_bootstraps = 200
max_evals = 30
scale = 'emot_func'

thresholds = {
    'phys_func': 83,
    'role_func': 58,
    'soc_func': 58,
    'emot_func': 71,
    'cog_func': 75,
    'pain': 25,
    'fatigue': 39,

            }

scale_type = {
    'pain': 'symptom',
    'fatigue': 'symptom',

}

#outcome = 'hn4_dv_c30_ghs'

outcome = 'hn4_dv_c30_'+scale
threshold = thresholds[scale] #
model_results_folder = 'logreg_'+scale

# check if directory exists
if os.path.isdir('results/'+model_results_folder):
    print(f"Warning: the directory results/{model_results_folder} already exists!")
    model_results_folder = model_results_folder  + '_2'
    print(f"Results will be saved in results/{model_results_folder} instead.")

# create directory
os.mkdir('results/'+model_results_folder)

variables_list_file = "data/prospective_variables.csv"

print(f"Modelling: hn4_dv_c30_{scale}")
print(f"Threshold: {threshold}")
print('Variables were selected from: '+variables_list_file)
print('Results will be saved at results/'+model_results_folder)


df = pd.read_csv("data/BD4QoL_030124_encoded.csv") # original data
bs = pd.read_csv("data/bootstrap_ids.csv")

# load variable list
variable_list = pd.read_csv(variables_list_file)

# clean the data
df = preprocess(data = df, features = variable_list, target = None)

df_orig = df.copy(deep=True)

# filter missing outcomes
df_orig = df_orig[~df_orig[outcome].isna()]

# define the independent variables
covariates = variable_list[variable_list['predictor']=='yes'].variable.values.tolist()

# separate predictors and outcome
X_orig = df_orig[covariates]

# define labels
df_orig = df_orig.assign(label = np.where(df_orig[outcome]<threshold, 1, 0))

y_orig = df_orig['label']

# Define the hyperparameter search space
space = {
    'alpha': hp.loguniform('alpha', -10, 0)

}

# Perform hyperparameter optimization on the original data
trials = Trials()
best = fmin(fn=partial(objective, X=X_orig, y=y_orig, n_splits = 5), 
            space=space, 
            algo=tpe.suggest, 
            max_evals=max_evals, 
            trials=trials)


# define model with best hyperparameters
model_best = LogisticRegression(
        penalty='l1', 
        solver='liblinear', 
        C = best['alpha'],
        max_iter = 10000
        )


# load preprocessor pipeline
preprocessor = load_preprocessor(X_orig)

# define pipeline
pipeline = Pipeline( steps = [
                    ('preprocessing', preprocessor),
                    ('regressor', model_best)]
                     )


pipeline.fit(X_orig, y_orig)

# save model
with open('results/' + model_results_folder + '/model_orig.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)


for bootstrap in range(0, n_bootstraps):
    print("Bootstrap:", bootstrap)
    # select observations in the bootstrap
    bs_n = df.iloc[bs[bs['bs_id'] == bootstrap + 1].loc[:, 'studyid'], : ].copy(deep=True)

    # filter missing outcomes
    bs_n = bs_n[~bs_n[outcome].isna()]

    bs_n = bs_n.assign(label = np.where(bs_n[outcome]<threshold, 1, 0))

    #  get out-of-bag observations
    # oob = get_oob_samples(df, bs_n.loc[:, "studyid_hn057"])
    
    # bootstrap data
    X_train = bs_n.loc[:, covariates]
    y_train = bs_n.loc[:, 'label']

    # bootstrap out-of-bag samples
    # X_oob = oob.loc[:, covariates]
    # y_oob = oob.loc[:, outcome]

    # hyperparametrization on the bootstrap
    # preprocessing is done inside the objective function
    # cross-validation loop to avoid data leakage
    trials = Trials()
    best = fmin(fn=partial(objective, X=X_train, y=y_train, n_splits = 5), 
            space=space, 
            algo=tpe.suggest, 
            max_evals=max_evals, 
            trials=trials)
    
    model_best =  LogisticRegression(
        penalty='l1', 
        solver='liblinear', 
        C = best['alpha'],
        max_iter = 10000
        )

    # load preprocesssing pipeline
    preprocessor = load_preprocessor(X_train)

    # define pipeline
    pipeline = Pipeline( steps = [
                    ('preprocessing', preprocessor),
                    ('regressor', model_best)]
                     )


    # fit the bootstrap model with optimum parameters
    pipeline.fit(X_train, y_train)

    # save model
    with open('results/' + model_results_folder + '/model_bs' + str(bootstrap + 1) + '.pkl', 'wb') as model_file:
        pickle.dump(pipeline, model_file)