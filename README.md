# BD4QoL
Big Data for Quality of Life (BD4QoL) EU project repository.


# Respository overview

- All model training and statistics are located in the folder `src`.
- `SLURM` shell scripts can be found in `jobs`.
- List of variables used for different analyses are found in `data`.

- Naming rules
  - Scripts used for data cleaning and preparation start with the prefix `A`, e.g. `A1_preprocessing.R`.
  - Scripts used to train models start with the letter `M`
  - Post-processing and statistics scripts start with the letter `E`
  - Functions and classes used in several other scripts are in `functions.py`


```
BD4QoL/
├── data/
│   ├── all_variables.csv
│   ├── prospective_variables.csv
│   ├── encoded_vars.csv
│   ├── propensity_variables.csv
│   └── bootstrap_ids.csv
├── src/
│   ├── __init__.py
│   ├── A1_preprocessing.R
│   ├── A2_bootstrap_resample.R
│   ├── E1_bootstrap_statistics.py
│   ├── E2_make_figures.py
│   ├── M1_logreg.py
│   ├── M2_xgboost.py
│   ├── M3_survival.py
│   ├── M4_xgboost_conformal.py
│   ├── M5_xgb_conformal_parallel.py
│   ├── M6_logreg_decline.py
│   ├── M7_lasso_conformal.py
│   ├── M8_lasso_conformal_parallel.py
│   ├── M9_survival_parallel.py
│   ├── M10_propensity.py
│   ├── M11_propensity_parallel.py
│   └── functions.py
├── jobs/
│   ├── M4_xgboost_conformal.sh
│   ├── M5_xgb_conformal_parallel_complete.sh
│   ├── M6_logreg_decline.sh
│   ├── M7_ols_lasso.sh
│   ├── M8_lasso_conformal_parallel.sh
│   ├── M5_commands.conf
│   └── M8_commands.conf
├── LICENSE
├── README.md
└── requirements.txt
```

## Requirements

```{txt}
scikit-learn==1.3.1
xgboost==2.0.0
pandas==2.1.0
seaborn==0.12.2
numpy==1.25.2
matplotlib==3.8.0
scipy==1.11.2
torch==2.0.1
jupyterlab
plotly
ipython
hyperopt==0.2.7
mapie==0.7.0
```


