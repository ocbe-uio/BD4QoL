from typing import Type, Dict
import sklearn
import pandas as pd
import numpy as np
import os
import pickle
import json

#from crps import crps_ensemble

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


def load_data(variables = 'reduced'):
    """
    load data, bootstrap indexes and variable list
    """

    if (variables == 'complete'):
        # use the complete set of variables
        variables_list_file = "data/all_variables.csv"
    else:
        # use the reduced set that matches with the prospective
        # study and favours usability
        variables_list_file = "data/prospective_variables.csv"
    
    df = pd.read_csv("data/BD4QoL_030124_encoded.csv") # original data
    bs = pd.read_csv("data/bootstrap_ids.csv")
    
    # load variable list
    variable_list = pd.read_csv(variables_list_file)
    
    # define the independent variables
    covariates = variable_list[variable_list['predictor']=='yes'].variable.values.tolist()

    # clean the data
    df = preprocess(data = df, features = variable_list, target = None)

    return df, bs, variable_list, covariates


def mkdir_safe(folder_name = '', prefix = 'results/'):
    '''
     check if directory exists and add the suffix _{i} 
    '''
    i = 1
    temp = folder_name
    while os.path.isdir(prefix + temp):
        i += 1
        print(f"Warning: the directory {prefix}{temp} already exists!")
        temp = folder_name  + '_' + str(i)
    
    os.mkdir('results/' + temp)
    print('Results will be saved at /results/' + temp)

    return temp 

def preprocess(data: Type[pd.DataFrame], 
              features: Type[pd.DataFrame], 
              target: str = None) -> Type[pd.DataFrame]:
    '''
    simple data preprocessing
        1. make all column names lower case
        2. make all categorical variables lower case
        3. subset the data
        4. add the correct types to the data frame
    :param data: data frame
    :param features: data frame with the variables list and enconding
    :param target: outcome variable name, if provided missing values are removed
    :return data: transformed data frame
    '''

    # make all variables lower case
    data.columns = map(str.lower, data.columns)

    # make all categories lower case
    data = data.apply(lambda x: x.str.lower() if(x.dtype == 'object') else x)

    # subset data
    data = data.loc[:, features.loc[:, "variable"]]

    # encode variables type according to variables list
    types = dict(zip(features.loc[:, "variable"], features.loc[:, "dtype"]))

    data = data.astype(types, errors='ignore')
    
    if target is not None:
        # filter missing outcomes
        data = data[~data[target].isna()]

    return data



def load_preprocessor(X: Type[pd.DataFrame], imputation_model: str = 'knn'):
    """
    preprocess data for neural network training
    Numerical transformations:
        - Impute with median
        - Standard scale
    Categorical transformations:
        - Impute with missing indicator
        - Encode with OneHotEncoder

    :param X: training data
    :return preprocessor: ColumnTransformer object
    
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer, MissingIndicator, KNNImputer

    # define the numerical and categorical features
    numeric_features = [col for col in X.columns if X[col].dtype != "category"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    # define the preprocessing steps
    # scale first and impute after since KNN works best with standardized features
    numeric_transformer = Pipeline(steps=[
        ('scale', StandardScaler()),
        ('num_imputer', KNNImputer(n_neighbors=5, weights="uniform"))
    ])

    # create the missing indicator first and encode after
    categorical_transformer = Pipeline(steps=[
        ('cat_imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='if_binary'))
        ])

    if imputation_model == 'knn':

        # apply the preprocessing steps to the features
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)])

    elif imputation_model == 'xgboost':
        # select this option if you want to 
        # use the implicit imputation performed by xgboost
        # Beware that leaving NaNs in the data
        # may be incompatible with some versions of hyperopt 
        # and sklearn Pipeline. 
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_features)],
                remainder='passthrough')

    #preprocessor = preprocessor.fit(X)
    
    return preprocessor


def imputation(data, model, target = None, index = None):
    '''
    impute the missing values in the data using the model pipeline
    and back-transform the data to retrieve them in the original scale

    Nomenclature:
    _in: original features
    _out: features after transformation

    '''
    
    num_features = model.named_steps['preprocessing'].named_transformers_['num'].get_feature_names_out()
    cat_features_in = model.named_steps['preprocessing'].named_transformers_['cat'].feature_names_in_
    cat_features_out = model.named_steps['preprocessing'].named_transformers_['cat'].get_feature_names_out()

    processed_data = model.named_steps['preprocessing'].transform(data)

    features_out = num_features.tolist() + cat_features_out.tolist()

    processed_data = pd.DataFrame(processed_data, columns = features_out)

    num_imputed = model.named_steps['preprocessing'].named_transformers_['num'].named_steps['scale'].inverse_transform(processed_data[num_features])
    cat_imputed = model.named_steps['preprocessing'].named_transformers_['cat'].named_steps['onehot'].inverse_transform(processed_data[cat_features_out])
    imputed = np.concatenate((num_imputed, cat_imputed), axis=1)

    features_in = num_features.tolist() + cat_features_in.tolist()

    imputed = pd.DataFrame(imputed, columns = features_in, index = data.index)

    if target is not None:
        imputed[target] = data[target]

    if index is not None:
        imputed[index] = data[index]

    return imputed


def kfold_indices(X: Type[np.ndarray], 
                  y: Type[np.ndarray] = None, 
                  n_splits: int = 5, 
                  shuffle: bool = False, 
                  random_state: int = None) -> Dict:
    """
    Extract the indices from the generator function KFold from sklearn as dictionary.
    
    :param X numpy.ndarray: features
    :param y numpy.ndarray: response
    :param n_splits int: number of folds
    :param shuffle bool: shuffle the data before splitting
    :param random_state: seed for the randoom generator
    :return folds dict: dictionary with folds indices

    Example:
    >>> X = np.random.rand(10,3)
    >>> y = np.random.rand(10)
    >>> kfold_indices(X, y)
        {'0': {'Train': array([2, 3, 4, 5, 6, 7, 8, 9]), 'Test': array([0, 1])},
         '1': {'Train': array([0, 1, 4, 5, 6, 7, 8, 9]), 'Test': array([2, 3])},
         '2': {'Train': array([0, 1, 2, 3, 6, 7, 8, 9]), 'Test': array([4, 5])},
         '3': {'Train': array([0, 1, 2, 3, 4, 5, 8, 9]), 'Test': array([6, 7])},
         '4': {'Train': array([0, 1, 2, 3, 4, 5, 6, 7]), 'Test': array([8, 9])}}
          
    """
    
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits= n_splits, shuffle=shuffle, random_state=random_state)
    
    folds = {}
    for i, (train_index, test_index) in enumerate(kf.split(X, y)):
        folds[str(i)] = {}
        folds[str(i)]['Train'] = train_index
        folds[str(i)]['Test'] = test_index

    return folds


def cross_validate(X, y, model, loss, n_splits = 5, random_state = None, weighting_function = None, **kwargs):
    """
    
    k-fold cross validation from scratch

    if the estimator is set to QuantileRegressor then you should pass quantile = 0.5 through **kwargs
    
    """
    
    folds = kfold_indices(X, y, n_splits = n_splits, shuffle = True, random_state = random_state)

    losses = np.zeros(n_splits)

    weights_train = None
    weights_test = None
    
    for  n in folds.keys():
        X_train = X.iloc[folds[str(n)]['Train']]
        y_train = y.iloc[folds[str(n)]['Train']]
        X_test = X.iloc[folds[str(n)]['Test']]
        y_test = y.iloc[folds[str(n)]['Test']]

        if weighting_function is not None:
            weights_train = weighting_function(y_train)
            weights_test = weighting_function(y_test)
        
        model.fit(X_train, y_train, regressor__sample_weight = weights_train)

        y_pred = model.predict(X_test)

        losses[int(n)] = loss(y_test, y_pred, sample_weight = weights_test, **kwargs)
        
    return losses


def get_oob_samples(original_data: Type[pd.DataFrame], bootstrap_case_ids: Type[np.ndarray]) -> Type[pd.DataFrame]:
    """
    :param original_data: dataframe with the original data
    :param bootstrap_case_ids: observation ids in the boostrap
    """
    oob = original_data[~original_data["studyid_hn057"].isin(bootstrap_case_ids)].copy(deep=True)
    
    return oob

def get_optimum_threshold(fpr, tpr, thresholds):
    i = np.arange(len(tpr)) # index for df
    roc = pd.DataFrame({
        'fpr' : pd.Series(fpr, index=i),
        'tpr' : pd.Series(tpr, index = i), 
        '1-fpr' : pd.Series(1-fpr, index = i), 
        'tf' : pd.Series(tpr - (1-fpr), index = i), 
        'thresholds' : pd.Series(thresholds, index = i)
        }
        )
    return roc.iloc[(roc.tf-0).abs().argsort()[:1]]

def dot632_estimator(theta_origin, theta_oob):
    """
    compute the .632 bootstrap estimator
    """
    return 0.368 * theta_origin + 0.632 * np.mean(theta_oob)





def __plot_calibration(y_true, y_pred, mode='percentile', show_hist = True):
    '''
    this function will not be required anymore and is deprecated
    '''
    
    import matplotlib.pyplot as plt
    from pygam import GAM, s, l

    if mode=='percentile' or mode=='both':
        percs = np.linspace(0,100,21)
    
        predicted_quantiles = np.percentile(y_pred, percs)
        sample_quantiles = np.percentile(y_true, percs)

        gam = GAM(s(0) + l(0), distribution = 'normal', verbose= False ).fit(predicted_quantiles, sample_quantiles)
        smooth = gam.predict(predicted_quantiles)
        ci = gam.confidence_intervals(predicted_quantiles)
    
        plt.plot(predicted_quantiles, ci, ls = '--', color = 'black')
        plt.plot(predicted_quantiles, smooth, color = 'red')    
    
        plt.scatter(predicted_quantiles, sample_quantiles, color='black', marker="o", s=5)
        
        x = np.linspace(np.min((predicted_quantiles.min(), sample_quantiles.min())), 
                    np.max((predicted_quantiles.max(), sample_quantiles.max())))

        plt.plot(x,x, color="k", ls=":", alpha=0.2)
        if show_hist:
            plt.hist(predicted_quantiles, alpha = 0.5)
        plt.ylabel('Sample quantiles')
        plt.xlabel('Predicted quantiles')
        plt.show()
        
    if mode=='raw' or mode=='both':
        x = np.linspace(np.min(y_true),np.max(y_true), 100)
        #preds = pd.DataFrame(np.concatenate((y_pred.reshape(-1,1), y_true.reshape(-1,1)), axis = 1), columns = ['y_pred', 'y_true'])
        
        gam = GAM(s(0) + l(0), distribution = 'normal', verbose= False ).fit(y_pred, y_true)
        smooth = gam.predict(x)
        ci = gam.confidence_intervals(x)

        plt.plot(x, ci, ls = '--', color = 'black')
        plt.plot(x, smooth, color = 'red')    
        plt.scatter(x = y_pred, y = y_true, s = 4.0, c = 'black', alpha=0.2)
        plt.ylabel('$Sample$')
        plt.xlabel('$Predicted$')
        plt.plot(x, x, linestyle = ':')

        plt.show()




def predict_wrapper(predict):
    '''
    Decorator to make the predict function general for binary and continuous outcomes
    '''
   
    def predict_general(X, **kwargs):
        prediction = predict(X, **kwargs)
        if prediction.ndim == 2:
            prediction = prediction[:, 1]

        return prediction

    return predict_general


def compute_bootstrap_metrics(n_bootstraps, 
                              df, 
                              bs, 
                              covariates, 
                              outcome, 
                              variable_id,
                              metrics_dict,
                              path = 'results/models/reduced/logreg_phys_func/',
                              threshold = None, 
                              outcome_type = 'binary', 
                              prediction_function = 'predict_proba',
                              json_file = None,
                              predict_args = {},
                              propensity_model = False,
                              crps = False
                              ):
    '''
    This function computes the bootstrap metrics for a given model and a given set of metrics
    provided by the metrics_dict. The metrics_dict is a dictionary with the name of the metric
    and the function that computes the metric. The function should have the following signature:
    metric_function(y_true, y_pred) where y_true is the true outcome and y_pred is the predicted
    outcome. The function should return a single value.

    // TODO:
        This function can be generalized further:
        - Allow metrics that take the classification instead of the probability

    '''

    #groups = ['overall']
    groups = {'overall': -1}

    df_orig = df.copy(deep=True)


    if propensity_model:
        # compute propensity scores for missing outcome
        with open('results/logreg_propensity/model_orig.pkl', 'rb') as file:
            scorer  = pickle.load(file)
        
        df_orig = df_orig.assign(missing = scorer.predict(df_orig))

        groups = {'overall': -1, 'missing=1': 1, 'missing=0':0}

    # filter missing outcomes
    df_orig = df_orig[~df_orig[outcome].isna()]


    # check if the outcome is binary or continuous
    if outcome_type == 'binary':
        #  define labels
        df_orig = df_orig.assign(label = np.where(df_orig[outcome]<threshold, 1, 0))
        y_orig = df_orig['label']
    
    if outcome_type == 'decline':
  
        df_orig = df_orig[~df_orig['hn3_dv_c30_ghs'].isna()]
        df_orig = df_orig.assign(decline =  np.where(df_orig['hn4_dv_c30_ghs']<=df_orig['hn3_dv_c30_ghs']-10, 1, 0))
        y_orig = df_orig.loc[:, 'decline']

    else:
 
        y_orig = df_orig[outcome]    

    # separate predictors and outcome
    X_orig = df_orig[covariates]
 

    bootstrap_metrics = {}

    model_file = path + 'model_orig.pkl'

    with open(model_file, 'rb') as file:
        model = pickle.load(file)


    # decorate the predict function to make it general for binary and continuous outcomes
    predict = predict_wrapper(getattr(model, prediction_function))

    # Make predictions on the original data
    y_pred_orig = predict(X_orig, **predict_args)

    
    df_orig = df_orig.assign(y_pred = y_pred_orig)
    

    for group, score in groups.items():
        print(group, score)
        if score >=0:
            y_pred_orig = df_orig[df_orig['missing'] == score ].loc[:, 'y_pred']

        for name, metric in metrics_dict.items():
            print(name)

            bootstrap_metrics[group] = {}

            bootstrap_metrics[group][name] = {}

            # Calculate the metrics for original data
            bootstrap_metrics[group][name]['orig'] = metric(y_orig, y_pred_orig)

            # train in the bootstrap and test in the oob samples
            bootstrap_metrics[group][name]['oob'] = n_bootstraps * [0]

            # train in the bootstrap and test in the bootstrap
            bootstrap_metrics[group][name]['bs'] = n_bootstraps * [0]

            # train in the bootstrap and test in the original data
            bootstrap_metrics[group][name]['bs_orig'] = n_bootstraps * [0]
    print(bootstrap_metrics['overall'])
    for bootstrap in range(n_bootstraps):
        print('Bootstrap', bootstrap)

        # load model from results/models/reduced
        model_file = path + 'model_bs' + str(bootstrap + 1) + '.pkl'

        with open(model_file, 'rb') as file:
            model = pickle.load(file)
        
        # decorate the predict function to make it general for binary and continuous outcomes
        predict = predict_wrapper(getattr(model, prediction_function))

        # select observations in the bootstrap
        bs_n = df.iloc[bs[bs['bs_id'] == bootstrap + 1].loc[:, 'studyid'], :].copy(deep=True)

        # filter missing outcomes
        bs_n = bs_n[~bs_n[outcome].isna()]

        # get the out-of-bag samples
        # Important: the OOB is being deep copied from the original data
        # it is important to filter missing outcomes in OOB afterwards
        # otherwise the performance will be falsily reduced by the NaNs in the data
        oob = get_oob_samples(df, bs_n.loc[:, variable_id])
    

        # check if the outcome is binary or continuous
        if outcome_type == 'binary':

            # define labels
            bs_n = bs_n.assign(label = np.where(bs_n[outcome] < threshold, 1, 0))
            oob = oob.assign(label = np.where(oob[outcome] < threshold, 1, 0))

            # select binary outcome
            y_train = bs_n.loc[:, 'label']
            y_oob = oob[~oob[outcome].isna()].loc[:, 'label']

        # elif outcome_type == 'decline':
        #     # define labels
        #     bs_n = bs_n.assign(label = np.where(bs_n['hn4_dv_c30_ghs'] <= bs_n['hn3_dv_c30_ghs']-10, 1, 0))
        #     oob = oob.assign(label = np.where(oob['hn4_dv_c30_ghs'] <= bs_n['hn3_dv_c30_ghs']-10, 1, 0))

        #     # select binary outcome
        #     y_train = bs_n.loc[:, 'label']
        #     y_oob = oob[~oob[outcome].isna()].loc[:, 'label']   
        elif outcome_type == 'decline':
            # filter missing baseline

            bs_n = bs_n[~bs_n['hn3_dv_c30_ghs'].isna()]
            bs_n = bs_n.assign(decline =  np.where(bs_n['hn4_dv_c30_ghs']<=bs_n['hn3_dv_c30_ghs']-10, 1, 0))

            oob = oob[~oob['hn3_dv_c30_ghs'].isna()]
            oob = oob[~oob['hn4_dv_c30_ghs'].isna()]
            oob = oob.assign(decline =  np.where(oob['hn4_dv_c30_ghs']<=oob['hn3_dv_c30_ghs']-10, 1, 0))

            y_train = bs_n.loc[:, 'decline']
            y_oob = oob.loc[:, 'decline']

        else:
            
            # select continuous outcome
            y_train = bs_n.loc[:, outcome]
            y_oob = oob[~oob[outcome].isna()].loc[:, outcome]

        # bootstrap data
        X_train = bs_n.loc[:, covariates]

        # bootstrap out-of-bag samples
        X_oob = oob[~oob[outcome].isna()].loc[:, covariates]


        # Make predictions on the test data
        predictions_oob = predict(X_oob, **predict_args)
        predictions_bs = predict(X_train, **predict_args)
        predictions_orig = predict(X_orig, **predict_args)

        for metric_name, metric_function in metrics_dict.items():
            # Calculate metrics
            bootstrap_metrics['overall'][metric_name]['oob'][bootstrap] = metric_function(
                 y_oob,  predictions_oob
                )
            bootstrap_metrics['overall'][metric_name]['bs'][bootstrap] = metric_function(
                y_train, predictions_bs
                )
            bootstrap_metrics['overall'][metric_name]['bs_orig'][bootstrap] = metric_function(
                 y_orig, predictions_orig
                )

        if crps:
            # this requires too much memmory >12Gb
            # the package is still in early development
            y_cdf_oob = model.named_steps['regressor'].get_cumulative_distribution_function(X_orig, predictions_oob)
            bootstrap_metrics['crps']['oob'][bootstrap] = crps_ensemble(y_oob, y_cdf_oob)

            y_cdf_bs = model.named_steps['regressor'].get_cumulative_distribution_function(X_orig, predictions_bs)
            bootstrap_metrics['crps']['bs'][bootstrap] = crps_ensemble(y_train, y_cdf_bs)

            y_cdf_orig = model.named_steps['regressor'].get_cumulative_distribution_function(X_orig, predictions_orig)
            bootstrap_metrics['crps']['bs_orig'][bootstrap] = crps_ensemble(y_orig, y_cdf_orig)



    
    if json_file is not None:
        with open(json_file, 'w') as file:
            json.dump(bootstrap_metrics, file, indent=4)

    return bootstrap_metrics



def slow_predict(model, X):
    '''
    this is a workaround because predict_proba is not 
    vectorized for the second argument `lower` and 
    there is no solution beyond vectorzing the class
    MapieConformalDistrution and re-training all the models.
    Explicit looping is slow, but the trade-off, considering
    the time and effort for the alternative solution, is worth it.
    '''
    y_pred_prob = np.zeros((X.shape[0]))
    n = 0
    for index, row in X.iterrows():

        y_pred_prob[n] = model.predict_proba(X.iloc[[n]], lower = X.loc[index,'hn3_dv_c30_ghs']-10.0 )
        n +=1 

    return y_pred_prob

def evaluate_decline_models(n_bootstraps, 
                              df, 
                              bs, 
                              covariates,  
                              variable_id,
                              metrics_dict,
                              outcome = 'hn4_dv_c30_ghs',
                              baseline = 'hn3_dv_c30_ghs',
                              propensity_model = False,
                              path = 'results/models/reduced/logreg_phys_func/',
                              json_file = None
                              ):
    '''
    This function computes the bootstrap metrics for a given model and a given set of metrics
    provided by the metrics_dict. The metrics_dict is a dictionary with the name of the metric
    and the function that computes the metric. The function should have the following signature:
    metric_function(y_true, y_pred) where y_true is the true outcome and y_pred is the predicted
    outcome. The function should return a single value.

    // TODO:
        This function be generalized further:
        - Allow metrics that take the classification instead of the probability

    '''

    model_file = path + 'model_orig.pkl'

    with open(model_file, 'rb') as file:
        model = pickle.load(file)


    df_orig = df.copy(deep=True)

    if propensity_model:
        # compute propensity scores for missing outcome
        with open('results/logreg_propensity/model_orig.pkl', 'rb') as file:
            scorer  = pickle.load(file)
        
        df_orig = df_orig.assign(missing = scorer.predict(df_orig))

    groups = {'overall': -1,  'missing=0': 0, 'missing=1': 1,}

    # filter missing outcomes
    df_orig = df_orig[~df_orig[outcome].isna()]
    df_orig = df_orig[~df_orig[baseline].isna()]


    # separate predictors and outcome
    X_orig = df_orig[covariates]

    #  define labels
    df_orig = df_orig.assign(label = np.where(df_orig[outcome] <= df_orig[baseline] - 10.0, 1, 0))

    y_orig = df_orig['label']

    bootstrap_metrics = {}


    # # Make predictions on the original data
    y_pred_orig = slow_predict(model, X_orig)


    df_orig = df_orig.assign(y_pred = y_pred_orig)
    
    for group, score in groups.items():

        if score >=0:
           y_pred_orig = df_orig[df_orig['missing'] == score ].loc[:, 'y_pred']
           y_orig = df_orig[df_orig['missing'] == score ].loc[:, 'label']
        else:
           y_pred_orig = df_orig.loc[:, 'y_pred']
           y_orig = df_orig.loc[:, 'label']

        for name, metric in metrics_dict.items():
            print(name)

            bootstrap_metrics[group] = {}

            bootstrap_metrics[group][name] = {}

            # Calculate the metrics for original data
            bootstrap_metrics[group][name]['orig'] = metric(y_orig, y_pred_orig)

            # train in the bootstrap and test in the oob samples
            bootstrap_metrics[group][name]['oob'] = n_bootstraps * [0]

            # train in the bootstrap and test in the bootstrap
            bootstrap_metrics[group][name]['bs'] = n_bootstraps * [0]

            # train in the bootstrap and test in the original data
            bootstrap_metrics[group][name]['bs_orig'] = n_bootstraps * [0]
    

    for bootstrap in range(n_bootstraps):
        print('Bootstrap', bootstrap)

        # load model from results/models/reduced
        model_file = path + 'model_bs' + str(bootstrap + 1) + '.pkl'

        with open(model_file, 'rb') as file:
            model = pickle.load(file)
    


        # select observations in the bootstrap
        bs_n = df.iloc[bs[bs['bs_id'] == bootstrap + 1].loc[:, 'studyid'], :].copy(deep=True)

        # filter missing outcomes
        bs_n = bs_n[~bs_n[outcome].isna()]
        bs_n = bs_n[~bs_n[baseline].isna()]

        if propensity_model:
            # compute propensity scores for missing outcome
            with open('results/logreg_propensity/model_bs' + str(bootstrap + 1) + '.pkl', 'rb') as file:
                scorer  = pickle.load(file)

            bs_n = bs_n.assign(missing = scorer.predict(bs_n))

        # impute predictors
        #bs_n = imputation(bs_n, model, outcome, variable_id)

        # get the out-of-bag samples
        # Important: the OOB is being deep copied from the original data
        # it is important to filter missing outcomes in OOB afterwards
        # otherwise the performance will be falsily reduced by the NaNs in the data
        oob = get_oob_samples(df, bs_n.loc[:, variable_id])
    

        # remove missing outcomes
        oob = oob[~oob[baseline].isna()]
        oob = oob[~oob[outcome].isna()]

        if propensity_model:
            oob = oob.assign(missing = scorer.predict(oob))

        # bootstrap data
        X_train = bs_n.loc[:, covariates]

        # define labels
        bs_n = bs_n.assign(label = np.where(bs_n[outcome] <= bs_n[baseline] - 10.0, 1, 0))
        oob = oob.assign(label = np.where(oob[outcome] <= oob[baseline] - 10.0, 1, 0))

        # select binary outcome
        y_train = bs_n.loc[:, 'label']
        y_oob = oob[~oob[outcome].isna()].loc[:, 'label']


        # bootstrap out-of-bag samples
        X_oob = oob[~oob[outcome].isna()].loc[:, covariates]


        # Make predictions on the test data
        predictions_oob = slow_predict(model, X_oob)
        predictions_bs = slow_predict(model, X_train)
        predictions_orig = slow_predict(model, X_orig)

        oob = oob.assign(y_pred = predictions_oob)
        bs_n = bs_n.assign(y_pred = predictions_bs)
        df_orig = df_orig.assign(y_pred = predictions_orig)

        y_oob = oob.loc[:, 'label']
        y_train = bs_n.loc[:, 'label']
        y_orig = df_orig.loc[:, 'label']

        for group, score in groups.items():

            if score >=0:
                predictions_oob = oob[oob['missing'] == score ].loc[:, 'y_pred']
                predictions_bs = bs_n[bs_n['missing'] == score ].loc[:, 'y_pred']
                predictions_orig = df_orig[df_orig['missing'] == score ].loc[:, 'y_pred']
                
                y_oob = oob[oob['missing'] == score ].loc[:, 'label']
                y_train = bs_n[bs_n['missing'] == score ].loc[:, 'label']
                y_orig = df_orig[df_orig['missing'] == score ].loc[:, 'label']


            for metric_name, metric_function in metrics_dict.items():
                # Calculate metrics
                bootstrap_metrics[group][metric_name]['oob'][bootstrap] = metric_function(
                    y_oob,  predictions_oob
                    )
                bootstrap_metrics[group][metric_name]['bs'][bootstrap] = metric_function(
                    y_train, predictions_bs
                    )
                bootstrap_metrics[group][metric_name]['bs_orig'][bootstrap] = metric_function(
                    y_orig, predictions_orig
                    )

     
    
    if json_file is not None:
        with open(json_file, 'w') as file:
            json.dump(bootstrap_metrics, file, indent=4)

    return bootstrap_metrics