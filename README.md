# BD4QoL
Big Data for Quality of Life (BD4QoL) EU project repository.


# Respository overview

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
