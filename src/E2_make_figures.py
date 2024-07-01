import pandas as pd
import numpy as np
from typing import Type
from sklearn import metrics
from sklearn import calibration 
import matplotlib.pyplot as plt
from functions import *

def bootstrapped_predictions(df, 
                             outcome, 
                             model_folder, 
                             bs, 
                             prediction_function = 'predict_proba', 
                             predict_args = {}, 
                             n_bootstraps = 200, 
                             verbose = True):

    import matplotlib.pyplot as plt
    import warnings 

    warnings.filterwarnings('ignore', message='.*Found unknown categories')

    preds = np.zeros((0, 3))

    if outcome == 'survival':
        # transform the outcome to binary
        df = df.assign(survival = np.where(df['hn4_dv_status'] == '2 - dead', 1, 0))

    # 200: 44
    for bootstrap in range(0, n_bootstraps):
        if bootstrap%10 == 0 and verbose:
            print(bootstrap)
        bs_n = df.iloc[bs[bs['bs_id'] == bootstrap + 1].loc[:, 'studyid'], : ].copy(deep=True)
        
        # filter missing outcomes
        bs_n = bs_n[~bs_n[outcome].isna()]
        
        # out-of-bag samples
        oob = get_oob_samples(df, bs_n.loc[:, "studyid_hn057"])
        
        # bootstrap data
        X_train = bs_n.loc[:, covariates]
        y_train = bs_n.loc[:, outcome]

        # bootstrap out-of-bag samples
        X_oob = oob[~oob[outcome].isna()].loc[:, covariates]
        y_oob = oob[~oob[outcome].isna()].loc[:, outcome]

        with open('results/' +model_folder+ '/model_bs'+str(bootstrap+1)+'.pkl', 'rb') as file:
            bs_model = pickle.load(file)
            
        # decorate the predict function to make it general for binary and continuous outcomes
        predict = predict_wrapper(getattr(bs_model, prediction_function))
    
        y_pred = predict(X_oob, **predict_args)
        
        preds = np.vstack((preds, np.vstack((y_oob, y_pred, np.ones(y_oob.shape[0])*(bootstrap+1))).T))


    preds = pd.DataFrame(preds, columns = ['y_true', 'y_pred', 'bs'])
 
    return preds


def bootstrapped_roc_curve(df, 
                           outcome, 
                           model_folder, 
                           bs, 
                           threshold,  
                           prediction_function = 'predict_proba', 
                           predict_args = {}, 
                           n_bootstraps = 200, 
                           verbose = True):
    import matplotlib.pyplot as plt
    import warnings 

    warnings.filterwarnings('ignore', message='.*Found unknown categories')

    preds = np.zeros((0, 3))
    auc_curve = np.zeros((0, 4))


    tprs = []
    aucs = []
    thresholds_ = []

    fpr_interp = np.linspace(0, 1, 100)
    # 200: 44
    for bootstrap in range(0, n_bootstraps):
        if bootstrap%10 == 0 and verbose:
            print(bootstrap)
        bs_n = df.iloc[bs[bs['bs_id'] == bootstrap + 1].loc[:, 'studyid'], : ].copy(deep=True)
        
        # filter missing outcomes
        bs_n = bs_n[~bs_n[outcome].isna()]
        
        # out-of-bag samples
        oob = get_oob_samples(df, bs_n.loc[:, "studyid_hn057"])
        
        # bootstrap data
        X_train = bs_n.loc[:, covariates]
        y_train = bs_n.loc[:, outcome]

        # bootstrap out-of-bag samples
        X_oob = oob[~oob[outcome].isna()].loc[:, covariates]
        y_oob = oob[~oob[outcome].isna()].loc[:, outcome]

        with open('results/' +model_folder+ '/model_bs'+str(bootstrap+1)+'.pkl', 'rb') as file:
            bs_model = pickle.load(file)
            
        # decorate the predict function to make it general for binary and continuous outcomes
        predict = predict_wrapper(getattr(bs_model, prediction_function))
        
        label = np.where(y_oob <= threshold, 1, 0)

        y_pred_prob = predict(X_oob, **predict_args)

        fpr, tpr, thresholds = metrics.roc_curve(label, y_pred_prob, pos_label=1, drop_intermediate=False)
        auc = metrics.roc_auc_score(label, y_pred_prob)
     
        interp_tpr = np.interp(fpr_interp, fpr, tpr)
        interp_thresholds = np.interp(fpr_interp, fpr, thresholds)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        thresholds_.append(interp_thresholds)
        aucs.append(auc)

        auc_curve = np.vstack((auc_curve, np.vstack((fpr, tpr, thresholds, np.ones(fpr.shape[0])*(bootstrap+1))).T))
        
        prob_true, prob_pred = calibration.calibration_curve(label, y_pred_prob, n_bins=10)

        preds = np.vstack((preds, np.vstack((prob_true, prob_pred, np.ones(prob_true.shape[0])*(bootstrap+1))).T))

    auc_curve = pd.DataFrame(auc_curve, columns = ['fpr', 'tpr', 'thresholds', 'bs'])
    preds = pd.DataFrame(preds, columns = ['y_true', 'y_pred', 'bs'])
 
    return (preds, auc_curve, tprs, aucs, thresholds_)

def plot_bootstrapped_roc_curve(aucs, auc_curve, tprs, thresholds, figure_name = None):

    mean_tpr = np.mean(tprs, axis=0)
    mean_thresholds = np.mean(thresholds, axis=0)
    fpr_interp = np.linspace(0, 1, 100)
    #mean_fpr = np.linspace(0, 1, len(mean_tpr))
    mean_tpr[-1] = 1.0
    mean_auc = metrics.auc(fpr_interp, mean_tpr)
    std_auc = np.std(aucs)

    #std_tpr = np.std(tprs, axis=0)
    auc_lower = np.percentile(aucs, 2.5, axis=0)
    auc_upper = np.percentile(aucs, 97.5, axis=0)

    tprs_lower = np.percentile(tprs, 2.5, axis=0) #np.minimum(mean_tpr + 1.96*std_tpr, 1)
    tprs_upper = np.percentile(tprs, 97.5, axis=0) #np.maximum(mean_tpr - 1.96*std_tpr, 0)

    
    
    optimum = get_optimum_threshold(fpr_interp, mean_tpr, mean_thresholds)


    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(
        fpr_interp,
        mean_tpr,
        color="black",
        label=r"Mean ROC (AUC = %0.2f [%0.2f, %0.2f])" % (mean_auc, auc_lower, auc_upper),
        lw=2,
        alpha=0.8,
    )



    std_tpr = np.std(tprs, axis=0)
    tprs_lower = np.minimum(tprs_lower, 1)
    tprs_upper = np.maximum(tprs_upper, 0)

    ax.fill_between(
        fpr_interp,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        #label=r"$\pm$ 2 std. dev.",
        label=r"$95\%$ Confidence Interval",
    )

    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        #title=f"Mean and Bootstrapped 95% Confidence Intervals",
    )

    ax.plot(mean_tpr, mean_tpr, ls=':', color='black')

    #cutoff = '(%.2f, %.2f, %.2f)' % (optimum['fpr'].to_numpy()[0], optimum['tpr'].to_numpy()[0], optimum['thresholds'].to_numpy()[0] )
    cutoff = 'Optimum cut-off = %.5f' %  optimum['thresholds'].to_numpy()[0]
    ax.scatter(optimum['fpr'].to_numpy()[0],  optimum['tpr'].to_numpy()[0], marker='x', color='red', s=100, label=cutoff, zorder=200) 
    #plt.text(x = 0.1 +optimum['fpr'].to_numpy()[0], y = optimum['tpr'].to_numpy()[0], s=cutoff)
    #ax.get_figure()
    ax.axis("square")
    ax.legend(loc="lower right")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[-4:], labels[-4:], loc="lower right")
    
    if figure_name is not None:
        
        plt.savefig(figure_name+'.svg', format = 'svg', dpi=300)
    # plt.show()

    plt.close()


def plot_calibration(preds: pd.DataFrame, 
                     y_label: str = 'GHS/QoL', 
                     ylims: tuple[int, int] = (0, 100), 
                     type: str = 'continuous',
                     figure_name: str = None,
                     n_bootstrap:int = 200,
                     plot_all: bool = False ) -> None:
    '''
    this function plots the calibration curve for continuous outcomes and probabilities
    change ylims to (0,1) for probabilities. 
    :param preds: expects a dataframe with y_true, y_pred and bs as columns
    '''
    from pygam import GAM, LinearGAM, s, l
    from sklearn import calibration 


    if type == 'probability':
        # overrid ylims if probability calibration
        ylims = (0, 1)


    x = np.linspace(ylims[0], ylims[1], 100)
    average = np.zeros(x.shape)

    all_smooth = []
    
    for bootstrap in range(1,n_bootstrap+1):

        
        if type == 'continuous':
            y_true, y_pred = (preds[preds['bs']==bootstrap].loc[:,'y_true'].to_numpy(), 
                              preds[preds['bs']==bootstrap].loc[:,'y_pred'].to_numpy() 
                             )
        elif type == 'probability':    
            y_true, y_pred = calibration.calibration_curve(
                preds[preds['bs']==bootstrap].loc[:,'y_true'].to_numpy(), 
                preds[preds['bs']==bootstrap].loc[:,'y_pred'].to_numpy(), 
                n_bins = 10
            )
        
        gam = LinearGAM(s(0, n_splines=8),  verbose= False ).fit(y_pred, y_true)   
        smooth = gam.predict(x)
        average = average + smooth/n_bootstrap
        all_smooth.append(smooth)
        #ci = gam.confidence_intervals(x)
        if plot_all:
            plt.plot(x, smooth, color = 'grey', alpha=0.2)  

    plt.plot(x, np.percentile(np.asarray(all_smooth), 2.5, axis = 0), color = 'grey', alpha=0.8 )
    plt.plot(x, np.percentile(np.asarray(all_smooth), 97.5, axis = 0),color = 'grey', alpha=0.8)
    plt.plot(x,x, color="k", ls=":", alpha=1.0, label = 'Perfect calibration')
    
    plt.plot(x, average, color = 'red', label  = 'Mean Bootstrap') 
    plt.xlabel('Predicted ' + y_label)
    plt.ylabel('Observed ' + y_label)
    plt.legend()

    if figure_name is not None:
    
        plt.savefig(figure_name+'.svg', format = 'svg', dpi=300)
        
    
    # plt.show()
    plt.close()

def bootstrapped_calibration_slopes(preds: pd.DataFrame, n_bootstraps: int = 200) -> np.ndarray:
    """
    fit calibration curves for bootstrapped predictions and return the slopes
    print the mean and 95% confidence intervals

    :param preds: data frame with the following columns
                y_true, y_pred, bs 
            where 
            `y_true` is the measured outcome
            `y_pred` is the predicted outcome
            `bs` is the resample index
    """

    from sklearn import linear_model

    
    calibration_slopes = np.zeros((n_bootstraps+1))
    calibration_intercepts = np.zeros((n_bootstraps+1))
    
    for bs in range(1,n_bootstraps+1):
    
        #print(bs)
        LM = linear_model.LinearRegression()
    
    
        LM.fit(preds[preds['bs']==bs].loc[:,'y_pred'].to_numpy().reshape(-1,1), preds[preds['bs']==bs].loc[:,'y_true'].to_numpy().reshape(-1,1))
        
        calibration_slopes[bs] = LM.coef_[0][0]
        calibration_intercepts[bs] = LM.intercept_[0]
        

    print("Mean Calibration Slope  | [95% CI]")
    print(calibration_slopes.mean(), np.percentile(calibration_slopes, 2.75), np.percentile(calibration_slopes, 97.5))

    print("Mean Calibration Intercept  | [95% CI]")
    print(calibration_intercepts.mean(), np.percentile(calibration_intercepts, 2.75), np.percentile(calibration_intercepts, 97.5))


def plot_prediction_intervals(y_true, y_pred, y_lower, y_upper) -> None:
    """
    plot prediction intervals accross the whole range of predictions
    and the actual measurements

    """
    import plotly.graph_objects as go

    fig = go.Figure()

    trace1 = go.Scatter(
        x = y_pred,
        y = y_pred,

       mode = 'markers',
                error_y = dict(
                    type = 'data',
                    symmetric = False,
                    array = y_upper,
                    arrayminus = y_lower),
        name = 'prediction'
        )
    
    trace2 = go.Scatter(
        x = y_pred,
        y = y_true,
        mode = 'markers',
        marker_size = 5,
        marker_symbol = 'circle',
        name = 'data',
        )

    fig.add_trace(trace1)
    fig.add_trace(trace2)
  
    fig.show()


def plot_prediction_bands(preds: Type[pd.DataFrame]) -> None:
    """
    plot prediction intervals with error bands accross the prediction range
    with plotly.

    :param preds DataFrame: A data frame with the following columns
                            y_pred: predicted value
                            y_true: observed true value 
                            y_lower: predicted lower 5% quantile 
                            y_upper: predicted upper 95% quantile
    """
    import plotly.graph_objects as go
    
    fig = go.Figure([
            go.Scatter(
                name='Measurement',
                x=preds['y_pred'],
                y=preds['y_true'],
                mode='markers+lines',
            line=dict(color='rgb(31, 119, 180)'),
            ),
            go.Scatter(
                name='Upper Bound',
                x=preds['y_pred'],
                y=preds['y_upper'],
                mode='lines',
                marker=dict(color="#444"),
                line=dict(width=0),
                showlegend=False
            ),
            go.Scatter(
                name='Lower Bound',
                x=preds['y_pred'],
                y=preds['y_lower'],
                marker=dict(color="#444"),
                line=dict(width=0),
                mode='lines',
                fillcolor='rgba(68, 68, 68, 0.3)',
                fill='tonexty',
                showlegend=False
            )
    ])
    fig.update_layout(
    yaxis_title='Prediction interval',
    xaxis_title='Prediction',
    hovermode="x"
    )
    fig.show()


if __name__ == '__main__':


    import warnings 

    warnings.filterwarnings('ignore', message='.*Found unknown categories')

    n_bootstraps = 200
    outcome = 'hn4_dv_c30_ghs'
    model_type = 'conformal_xgboost'
    scale_threshold = 83
    scale = 'ghs'
    interps = []
    variables = 'complete'

    if (variables == 'complete'):
        # use the complete set of variables
        model_folder = 'full_'+ model_type + '_' + scale
        variables_list_file = "data/all_variables.csv"
    else:
        # use the reduced set that matches with the prospective
        # study and favours usability
        model_folder ='reduced_' + model_type + '_' + scale
        variables_list_file = "data/prospective_variables.csv"

    df = pd.read_csv("data/BD4QoL_030124_encoded.csv") # original data
    bs = pd.read_csv("data/bootstrap_ids.csv")

    # load variable list
    variable_list = pd.read_csv(variables_list_file)

    # define the independent variables
    covariates = variable_list[variable_list['predictor']=='yes'].variable.values.tolist()

    # clean the data
    df = preprocess(data = df, features = variable_list, target = None)

    print("Probability model calibration")
    # probability calibration
    # compute the predictions from OOB 
    preds  = bootstrapped_predictions(df = df, 
                                    outcome = outcome, 
                                    model_folder = model_folder, 
                                    bs = bs, 
                                    prediction_function = 'predict_proba',
                                    predict_args = {'lower': scale_threshold},
                                    n_bootstraps = n_bootstraps)


    preds_prob = preds.assign(y_true = np.where(preds['y_true'] <= scale_threshold, 1.0, 0.0))
    preds_prob = preds_prob.astype(dtype = 'float64')


    plot_calibration(
                    preds_prob, 
                    y_label ='Low QoL/GHS Probability', 
                    type = 'probability', 
                    n_bootstrap = n_bootstraps, 
                    figure_name = 'plots/calib_prob_' + model_folder
                    )

    print("Continuous model calibration")
    # continuous model calibration

    # compute oob predictions
    preds_continuous  = bootstrapped_predictions(df = df, 
                                    outcome = outcome, 
                                    model_folder = model_folder, 
                                    bs = bs, 
                                    prediction_function = 'predict',
                                    predict_args = {},
                                    n_bootstraps = n_bootstraps)


    plot_calibration(
                    preds_continuous, 
                    y_label ='QoL/GHS', 
                    type = 'continuous', 
                    n_bootstrap = n_bootstraps, 
                    figure_name = 'plots/calib_conti_' + model_folder
                    )

    print("Receiving Operator Curve")
    # compute ROCs
    preds, auc_curve, tprs, aucs, thresholds = bootstrapped_roc_curve(
                                                                    df = df, 
                                                                    outcome = outcome, 
                                                                    model_folder = model_folder, 
                                                                    bs = bs, 
                                                                    threshold = scale_threshold, 
                                                                    prediction_function = 'predict_proba',
                                                                    predict_args = {'lower': scale_threshold},
                                                                    n_bootstraps = n_bootstraps
                                                                    )

    # plot roc
    plot_bootstrapped_roc_curve(
                                aucs, 
                                auc_curve,tprs, 
                                thresholds, 
                                figure_name = 'plots/roc_'+model_folder
                                )