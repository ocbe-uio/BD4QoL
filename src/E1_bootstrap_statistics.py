"""
In this script we load the models trained with bootstrap data
and compute some statistics such as performance metrics.
"""


import pandas as pd
from functions import *
import warnings 

warnings.filterwarnings('ignore', message='.*Found unknown categories')
#import pickle


# import metrics form sklearn
from sklearn import metrics

metrics_dict = {
     #'r2_score': r2_score, 
     #'mse': mean_squared_error
    'brier': metrics.brier_score_loss,
    'roc_auc': metrics.roc_auc_score,
    #'accuracy': metrics.accuracy_score,
    #'precision': metrics.precision_score,
    #'recall': metrics.recall_score,
    #'f1': metrics.f1_score
} 



outcome = 'hn4_dv_c30_ghs'
scale = 'ghs'
#models_root_directory = 'results/models/reduced/'
#model_directory = 'logreg_phys_func/'  

#outcome_type = 'binary'

n_bootstraps = 200
outcome = 'hn4_dv_c30_ghs'
model_type = 'conformal_lasso'
model_type = 'logreg_survival'
model_type = 'logreg_decline'
scale_threshold = 83
scale = '_ghs'
scale = ''
interps = []
variables = 'reduced' #'propensity'
outcome = 'decline' # 'missing'
outcome = 'survival'
outcome = 'decline_classic'

if (variables == 'complete'):
    # use the complete set of variables
    model_folder = 'full_'+ model_type  + scale
    variables_list_file = "data/all_variables.csv"
elif (variables == 'propensity'):
    variables_list_file = "data/propensity_variables.csv"
    model_folder = 'logreg_propensity'
else:
    # use the reduced set that matches with the prospective
    # study and favours usability
    model_folder ='reduced_' + model_type + scale
    variables_list_file = "data/prospective_variables.csv"




#outcome = 'hn4_dv_c30_phys_func'
#models_root_directory = 'results/models/reduced/'
#model_directory = 'logreg_phys_func/'  
#threshold = 83
n_bootstraps = 200
#outcome_type = 'binary'

df = pd.read_csv("data/BD4QoL_030124_encoded.csv", sep=',') # original data
bs = pd.read_csv("data/bootstrap_ids.csv") # bootstrap ids


# load variable list
variable_list = pd.read_csv(variables_list_file)

# define the independent variables
covariates = variable_list[variable_list['predictor']=='yes'].variable.values.tolist()


# clean the data
df = preprocess(data = df, features = variable_list, target = None)

scale = 'phys_func'

thresholds = {
    'phys_func': 83,
    'role_func': 58,
    'soc_func': 58,
    'emot_func': 71,
    'cog_func': 75,
    #'pain': 25,
    #'fatigue': 39,

            }



# for scale in thresholds.keys():
    
#     print(scale)

#     outcome = 'hn4_dv_c30_'+scale
#     model_folder= 'logreg_' + scale 
#     threshold = thresholds[scale]

#     results = compute_bootstrap_metrics(n_bootstraps = 200,
#                             df = df,
#                             bs = bs,
#                             covariates = covariates,
#                             outcome = outcome,
#                             variable_id = 'studyid_hn057',
#                             metrics_dict = metrics_dict,
#                             path = 'results/reduced/'+model_folder+'/',
#                             threshold = threshold,
#                             outcome_type = 'binary',
#                             prediction_function = 'predict_proba',
#                             json_file = 'results/reduced/'+model_folder+'.json',
#                             predict_args = {}
#                             )



# metrics_dict = {
#      'r2_score': metrics.r2_score, 
#      'mse': metrics.mean_squared_error,
#      'mae': metrics.mean_absolute_error
#     #'brier': metrics.brier_score_loss,
#     #'roc_auc': metrics.roc_auc_score,
#     #'accuracy': metrics.accuracy_score,
#     #'precision': metrics.precision_score,
#     #'recall': metrics.recall_score,
#     #'f1': metrics.f1_score
# } 



# results = compute_bootstrap_metrics(n_bootstraps = 200,
#                             df = df,
#                             bs = bs,
#                             covariates = covariates,
#                             outcome = outcome,
#                             variable_id = 'studyid_hn057',
#                             metrics_dict = metrics_dict,
#                             path = 'results/'+model_folder+'/',
#                             threshold = threshold,
#                             outcome_type = 'continuous',
#                             prediction_function = 'predict',
#                             json_file = 'results/'+model_folder+'.json',
#                             crps = False
#                             )


metrics_dict = {
    #'brier': metrics.brier_score_loss,
    'roc_auc': metrics.roc_auc_score,
    #'accuracy': metrics.accuracy_score,
    #'precision': metrics.precision_score,
    #'recall': metrics.recall_score,
    #'f1': metrics.f1_score
} 


if outcome =='missing':
    df['hn4_dv_c30_ghs'] = df['hn4_dv_c30_ghs'].fillna(value=-100)

    # transform the outcome to binary
    df = df.assign(label = np.where(df['hn4_dv_c30_ghs']<0, 1, 0))
   
    results = compute_bootstrap_metrics(
                                n_bootstraps = 0, 
                                df = df, 
                                bs = bs, 
                                covariates = covariates,  
                                variable_id = 'studyid_hn057',
                                metrics_dict = metrics_dict,
                                prediction_function = 'predict_proba',
                                outcome_type = 'other',
                                propensity_model = True,
                                outcome ='label',
                                path = 'results/'+model_folder+'/',
                                json_file = 'results/'+model_folder+'propensity.json'
    )

elif outcome == 'decline':

    results = evaluate_decline_models(
                                    n_bootstraps = 200, 
                                    df = df, 
                                    bs = bs, 
                                    covariates = covariates,  
                                    variable_id = 'studyid_hn057',
                                    metrics_dict = metrics_dict,
                                    outcome = 'hn4_dv_c30_ghs',
                                    baseline = 'hn3_dv_c30_ghs',
                                    path = 'results/'+model_folder+'/',
                                    propensity_model = True,
                                    json_file = 'results/'+model_folder+'_decline_propensity.json'
    )
elif outcome == "survival":

    df = df.assign(survival=np.where(df['hn4_dv_status'] == '2 - dead', 1, 0))
    results = compute_bootstrap_metrics(n_bootstraps = 200,
                                        df = df,
                                        bs = bs,
                                        covariates = covariates,
                                        outcome = 'survival',
                                        variable_id = 'studyid_hn057',
                                        metrics_dict = metrics_dict,
                                        path = 'results/'+model_folder+'/',
                                        threshold = scale_threshold,
                                        outcome_type = 'survival',
                                        prediction_function = 'predict_proba',
                                        json_file = 'results/'+model_folder+'.json',
                                        predict_args = {},
                                        crps = False
    )
elif outcome == 'decline_classic':
    #

    #df = df.assign(decline = np.where(~np.isnan(df['hn4_dv_c30_ghs']) & ~np.isnan(df['hn3_dv_c30_ghs']),
    #np.where(df['hn4_dv_c30_ghs']<=df['hn3_dv_c30_ghs']-10, 1, 0), np.nan ))

    results = compute_bootstrap_metrics(n_bootstraps = 200,
                                        df = df,
                                        bs = bs,
                                        covariates = covariates,
                                        outcome = 'hn4_dv_c30_ghs',
                                        variable_id = 'studyid_hn057',
                                        metrics_dict = metrics_dict,
                                        path = 'results/'+model_folder+'/',
                                        threshold = scale_threshold,
                                        outcome_type = 'decline',
                                        prediction_function = 'predict_proba',
                                        json_file = 'results/'+model_folder+'.json',
                                        predict_args = {},
                                        crps = False
    )