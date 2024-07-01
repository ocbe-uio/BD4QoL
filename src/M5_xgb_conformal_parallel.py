import sys
print('This script is inteded to be used in parallel with taskfarm.')
print('We parallelize the training in each bootstrap batch.')
print('The use must provide the interval of bootstraps that will be computed.')

# Check if arguments are provided

if len(sys.argv) != 3:
    print(sys.argv)
    print(len(sys.argv))
    print("Usage: python script.py <first bootstrap index> <last bootstrap index>")
    
    sys.exit()

start_bootstrap = int(sys.argv[1])
end_bootstrap = int(sys.argv[2])

print(f'Computing bootstraps between {start_bootstrap} and {end_bootstrap}.')

from sklearn.pipeline import Pipeline
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from functools import partial
import pandas as pd
import numpy as np
import os
from hyperopt import fmin, tpe, hp, Trials
import pickle


from functions import *
from mapie.conformity_scores import AbsoluteConformityScore
from mapie.regression import MapieRegressor


class MapieConformalPredictiveDistribution(MapieRegressor):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.conformity_score.sym = False

    def get_cumulative_distribution_function(self, X, y_pred):
        '''
        this is based on the paper: 
        "Nonparametric predictive distributions based on conformal prediction" (2017) 
        Vovk et al 

        get_estimation_distribution() computes the Equation (22) in the paper
            C_i = \hat{y}_{n+1} + (y_i + \hat{y}_i) 
            np.add(y_hat, conformity_scores)
        then it can be sorted in increasing order to obtain the predictive distribution
        '''
        
        cs = self.conformity_scores_[~np.isnan(self.conformity_scores_)]

        res = self.conformity_score_function_.get_estimation_distribution(
            X, y_pred.reshape((-1, 1)), cs
        )
        return res

    def find_nearest(self, array, value):
        '''
        find the closest value in array
        '''
        array = np.asarray(array)
        value = np.asarray(value)
        idx = (np.abs(array - value.reshape(-1,1))).argmin(axis=1)
        return idx[0]


    def predict_proba(self, X, lower = None, upper = None):
        y_pred = self.predict(X)
        y_cdf = self.get_cumulative_distribution_function(X, y_pred)
        probability = np.zeros((X.shape[0]))
        
        for observation in range(X.shape[0]):
            counts, bins = np.histogram(y_cdf[observation], bins=100)
            cdf = np.cumsum(counts)/np.sum(counts)
        
            if lower is not None:
                #indices = self.find_nearest(bins, lower).reshape(-1,1)
    
                probability[observation] =  cdf[self.find_nearest(bins, lower)-1] #np.take_along_axis(cdf, indices = indices, axis = 1)
                #probability =  cdf[ self.find_nearest(bins, upper) ] - cdf[ self.find_nearest(bins, lower) ]
            else:
                probability = cdf[ self.find_nearest(bins, y_pred) -1 ]
            
        return probability

    def predict_class(self, X, lower, threshold = 0.5):

        probability = self.predict_proba(X, lower = lower)

        return np.where(probability > threshold, 1, 0)


def objective(params, X, y, n_splits):
    """
    this is the objective function that will be minimized by hyperopt
    to estimate the best hyperparameters for xgboost
    """
    # load preprocessing pipeline
    # in this case we only need OneHotEncoder
    # imputation will be handled by xgboost
    preprocessor = load_preprocessor(X, imputation_model = 'knn')

    model = MapieConformalPredictiveDistribution(
        estimator = XGBRegressor(
            n_estimators = int(params['n_estimators']),
            max_depth = int(params['max_depth']),
            gamma = params['gamma'],
            learning_rate = params['learning_rate'],
            subsample = params['subsample'],
            colsample_bytree = params['colsample_bytree'],
            min_child_weight = params['min_child_weight'],
            eta = params['eta'],
            random_state=42),
        conformity_score = AbsoluteConformityScore(),
        method = 'plus',
        random_state = 42,
        n_jobs = 1
    )
    
    # define pipeline
    pipeline = Pipeline( steps = [
                    ('preprocessing', preprocessor),
                    ('regressor', model)]
                     )

    losses = cross_validate(X, y, model = pipeline, loss = metrics.mean_absolute_error, n_splits = n_splits)

    mean_loss = np.mean(losses)
    
    return mean_loss

variables = "reduced" # complete set of variables or subset?

max_evals = 30
scale = 'ghs'
outcome = 'hn4_dv_c30_'+scale
model_type = 'parallel_conformal_xgboost_'

if (variables == 'complete'):
    # use the complete set of variables
    model_results_folder = 'full_'+ model_type + scale
    variables_list_file = "data/all_variables.csv"
else:
    # use the reduced set that matches with the prospective
    # study and favours usability
    model_results_folder = 'reduced_' + model_type + scale
    variables_list_file = "data/prospective_variables.csv"


# create directory
#model_results_folder = mkdir_safe(folder_name = model_results_folder)   

print('Variables were selected from: '+variables_list_file)

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
y_orig = df_orig[outcome]

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


#missing_numbers = np.arange(0,200)


for i_bootstrap in range(start_bootstrap, end_bootstrap):
    bootstrap = i_bootstrap # missing_numbers[i_bootstrap] - 1

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
    
    model_best = MapieConformalPredictiveDistribution(
        estimator = XGBRegressor(
            n_estimators = int(best['n_estimators']),
            max_depth = int(best['max_depth']),
            gamma = best['gamma'],
            learning_rate = best['learning_rate'],
            subsample = best['subsample'],
            colsample_bytree = best['colsample_bytree'],
            min_child_weight = best['min_child_weight'],
            eta = best['eta'],
            random_state=42),
        conformity_score = AbsoluteConformityScore(),
        method = 'plus',
        random_state = 42
    )
    

    # load preprocesssing pipeline, in this case only OneHotEncoder is needed
    preprocessor = load_preprocessor(X_train, imputation_model = 'knn')

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