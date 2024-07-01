import pandas as pd
import numpy as np
from functools import partial
from sklearn import metrics, linear_model, model_selection 
from sklearn.pipeline import Pipeline


import json
from functions import *

start_bootstrap = 0 
end_bootstrap = 200

outcome = 'missing'
model_type = 'logreg_propensity'

model_results_folder = model_type 

variables_list_file = "data/propensity_variables.csv"


# create directory
model_results_folder = mkdir_safe(folder_name = model_results_folder)   

print('Variables were selected from: '+variables_list_file)

df = pd.read_csv("data/BD4QoL_030124_encoded.csv") # original data
bs = pd.read_csv("data/bootstrap_ids.csv")

# load variable list
variable_list = pd.read_csv(variables_list_file)

# clean the data
df = preprocess(data = df, features = variable_list, target = None)

df['hn4_dv_c30_ghs'] = df['hn4_dv_c30_ghs'].fillna(value=-100)

# transform the outcome to binary
df = df.assign(missing = np.where(df['hn4_dv_c30_ghs']<0, 1, 0))

df_orig = df.copy(deep=True)

df_orig = df_orig[ df_orig['hn4_dv_status']!='2 - dead']

# define the independent variables
covariates = variable_list[variable_list['predictor']=='yes'].variable.values.tolist()

# separate predictors and outcome
X_orig = df_orig[covariates]
y_orig = df_orig[outcome]

# define model with best hyperparameters
model = linear_model.LogisticRegression()


# load preprocessor pipeline
preprocessor = load_preprocessor(X_orig)

# define pipeline
pipeline = Pipeline( steps = [
                    ('preprocessing', preprocessor),
                    ('regressor', model)]
                     )


pipeline.fit(X_orig, y_orig)

# save model
with open('results/' + model_results_folder + '/model_orig.pkl', 'wb') as model_file:
    pickle.dump(pipeline, model_file)


for bootstrap in range(start_bootstrap, end_bootstrap):
    print("Bootstrap:", bootstrap)
    # select observations in the bootstrap
    bs_n = df.iloc[bs[bs['bs_id'] == bootstrap + 1].loc[:, 'studyid'], : ].copy(deep=True)

    # filter deceased patients at follow up
    bs_n = bs_n[ bs_n['hn4_dv_status']!='2 - dead']
  

    # bootstrap data
    X_train = bs_n.loc[:, covariates]
    y_train = bs_n.loc[:, outcome]

    
    model_best = linear_model.LogisticRegression()

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