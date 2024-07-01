import pandas as pd
import numpy as np
from functools import partial
from sklearn import metrics, linear_model, model_selection 
from sklearn.pipeline import Pipeline
from hyperopt import fmin, tpe, hp, Trials

import json
from functions import *



def objective(params, X, y, n_splits):
    """
    this is the objective function that will be minimized by hyperopt
    to estimate the best hyperparameters, in this case only the l1-penalizaton alpha
    """
    # load preprocessor pipeline
    preprocessor = load_preprocessor(X)

    model = linear_model.LogisticRegression(
                        penalty = 'l1', 
                        C = params['alpha'], 
                        solver ='saga', 
                        max_iter=100000
                    )

    # define pipeline
    pipeline = Pipeline( steps = [
                    ('preprocessing', preprocessor),
                    ('regressor', model)]
                     )

    #losses = cross_validate(X, y, model = pipeline, loss = metrics.roc_auc_score, n_splits = n_splits)
    losses = model_selection.cross_val_score(pipeline, X, y, cv = n_splits, scoring = 'roc_auc')
    mean_loss = -np.mean(losses)
    
    return mean_loss



variables = "complete" # complete set of variables or subset?
start_bootstrap = 0 
end_bootstrap = 0
max_evals = 30

outcome = 'survival'
model_type = 'logreg_survival'

if (variables == 'complete'):
    # use the complete set of variables
    model_results_folder = 'full_'+ model_type 
    variables_list_file = "data/all_variables.csv"
else:
    # use the reduced set that matches with the prospective
    # study and favours usability
    model_results_folder = 'reduced_' + model_type 
    variables_list_file = "data/prospective_variables.csv"


# create directory
model_results_folder = mkdir_safe(folder_name = model_results_folder)   

print('Variables were selected from: '+variables_list_file)

df = pd.read_csv("data/BD4QoL_030124_encoded.csv") # original data
bs = pd.read_csv("data/bootstrap_ids.csv")



# load variable list
variable_list = pd.read_csv(variables_list_file)

# clean the data
df = preprocess(data = df, features = variable_list, target = None)

# transform the outcome to binary
df = df.assign(survival=np.where(df['hn4_dv_status'] == '2 - dead', 1, 0))

df_orig = df.copy(deep=True)

# filter missing outcomes
df_orig = df_orig[~df_orig[outcome].isna()]

# define the independent variables
covariates = variable_list[variable_list['predictor']=='yes'].variable.values.tolist()

# separate predictors and outcome
X_orig = df_orig[covariates]
y_orig = df_orig[outcome]

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
model_best = linear_model.LogisticRegression(
                    penalty = 'l1', 
                    C = best['alpha'], 
                    solver ='saga', 
                    max_iter=1000
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


for bootstrap in range(start_bootstrap, end_bootstrap):
    print("Bootstrap:", bootstrap)
    # select observations in the bootstrap
    bs_n = df.iloc[bs[bs['bs_id'] == bootstrap + 1].loc[:, 'studyid'], : ].copy(deep=True)

    # filter missing outcomes
    bs_n = bs_n[~bs_n[outcome].isna()]

    # bootstrap data
    X_train = bs_n.loc[:, covariates]
    y_train = bs_n.loc[:, outcome]

    # hyperparametrization on the bootstrap
    # preprocessing is done inside the objective function
    # cross-validation loop to avoid data leakage
    trials = Trials()
    best = fmin(fn=partial(objective, X=X_train, y=y_train, n_splits = 5), 
            space=space, 
            algo=tpe.suggest, 
            max_evals=max_evals, 
            trials=trials)
    
    model_best = linear_model.LogisticRegression(
                    penalty = 'l1', 
                    C = best['alpha'], 
                    solver ='saga', 
                    max_iter=100000
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