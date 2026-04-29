"""
This file is adapted from the Cause–Trigger algorithm code accompanying:
Hlaváčková-Schindler, K., Wöß, R., Pecorino, V., & Schindler, P. (2025).
"Cause or Trigger? From Philosophy to Causal Modeling."
Zenodo. DOI: 10.5281/zenodo.15109084
"""

from distfit import distfit
from statsmodels.tsa.api import VAR
"""
Extracts parameters which are necessary for the causal inference algorithms, aka the lag and distribution
"""
def select_var_lag(df, max_lags=2, criterion='aic'):
    """
    Selects the optimal lag for a multivariate time series using VAR.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame with multivariate time series data.
        max_lags (int): Maximum number of lags to consider.
        criterion (str): Criterion to minimize ('aic', 'bic', 'fpe', 'hqic').
        
    Returns:
        int: The best lag length for the VAR model.
    """
    model = VAR(df)
    result = model.select_order(maxlags=max_lags)
    return getattr(result, criterion)


def find_distribution(series):
    dist = distfit(distr=['gamma','invgauss','norm'], random_state=0, verbose=False)
    dv = dist.fit_transform(series.values)
    best_distribution = dv['model']['name']
    return convert_name(best_distribution)

def convert_name(name:str): 
     convert = {
          "gamma": "gamma", 
          "norm": "gaussian", 
          "invgauss":"inverse_gaussian"
     }
     return convert[name]

#def find_parameters(series):
#    distribution = find_distribution(series)
#    lag = select_var_lag(series)
#    return distribution, lag

def find_parameters(X, target_series, max_lags=2, criterion="aic"):
    lag = select_var_lag(X, max_lags=max_lags, criterion=criterion)
    distribution = find_distribution(target_series)
    return distribution, lag