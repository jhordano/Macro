# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 18:49:06 2019
@author: s3179575
"""

 # some example data
import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.tsa.api import VAR, DynamicVAR
mdata = sm.datasets.macrodata.load_pandas().data

 # prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)

quarterly = dates["year"] + "Q" + dates["quarter"]

from statsmodels.tsa.base.datetools import dates_from_str

quarterly = dates_from_str(quarterly)

mdata = mdata[['realgdp','realcons','realinv']]

mdata.index = pd.DatetimeIndex(quarterly)

data = np.log(mdata).diff().dropna()

 # make a VAR model
model = VAR(data)

results = model.fit(1)
print(results.summary())
# %%
results.plot()
# Plotting time series autocorrelation function:
results.plot_acorr()

# one can pass a maximum number of lags and the order criterion to use for order selection:
results = model.fit(maxlags=15, ic='aic')

irf = results.irf(10)
irf.plot(orth=False)


# %%
#  Question 1
from numpy.linalg import inv

def VAR_OLS(data, lags):
    lags = lags
    T,k = data.shape
    Y = data.iloc[range(lags,T)]    
    X = pd.DataFrame([])
    
    for i in range(lags,0,-1):
       # print(i)
        X_k = data.iloc[range(i-1,T-1-(lags-i))].reset_index(drop=True)
        X_k.rename(columns=lambda x: x +'_' +str(lags+1-i), inplace=True)
        X = pd.concat([X,X_k],axis=1)
        
    X['c'] = 1 
    T = T-lags
    Y = np.matrix(Y)
    X = np.matrix(X)
    XtX = np.matmul(np.transpose(X),X)
    beta = np.matmul(inv(XtX),np.matmul(np.transpose(X),Y))
    e = Y - np.matmul(X,beta)
    sigma_ols = (1/(T-k*lags-1))*np.matmul(np.transpose(e),e)
    var_b = np.kron(XtX,sigma_ols)
    sigma_ml = (1/T)*np.matmul(np.transpose(e),e)
    V_p = np.log(np.linalg.det(sigma_ml))
    AIC = V_p + (2/T)*(lags*np.power(k,2)+k)
    BIC = V_p + (np.log(T)/T)*(lags*np.power(k,2)+k)
    lag_sel = pd.DataFrame(columns=['AIC','BIC'])
    lag_sel = lag_sel.append({'AIC':AIC,'BIC':BIC,},ignore_index=True)
    return beta, lag_sel

beta_1_PE,sigma = VAR_OLS(data,1)

# %%
beta_1_PR = VAR_OLS(data,1)
beta_1_PT = VAR_OLS(data,1)

# Question 2
# Test for granger causality


# %%
# Question 3
# VAR model for p=1,2,3,4
import matplotlib.pyplot as plt
Lag_selec = pd.DataFrame(columns=['AIC','BIC'])
for i in range(1,5):
    print(i)
    beta_1_PE,Lag_s = VAR_OLS(data,i)
    Lag_selec = pd.concat([Lag_selec,Lag_s],axis=0)
    
Lag_selec = Lag_selec.reset_index(drop=True)    
plt.plot(Lag_selec.index,Lag_selec.iloc[:,0])
plt.plot(Lag_selec.index,Lag_selec.iloc[:,1]) 

# %%
# Question 5
def VAR_OLS(data, lags):
    lags = lags
    T,k = data.shape
    Y = data.iloc[range(lags,T)]    
    X = pd.DataFrame([])
    
    for i in range(lags,0,-1):
       # print(i)
        X_k = data.iloc[range(i-1,T-1-(lags-i))].reset_index(drop=True)
        X_k.rename(columns=lambda x: x +'_' +str(lags+1-i), inplace=True)
        X = pd.concat([X,X_k],axis=1)
        
    X['c'] = 1 
    T = T-lags
    Y = np.matrix(Y)
    X = np.matrix(X)
    XtX = np.matmul(np.transpose(X),X)
    beta = np.matmul(inv(XtX),np.matmul(np.transpose(X),Y))
    e = Y - np.matmul(X,beta)
    sigma_ols = (1/(T-k*lags-1))*np.matmul(np.transpose(e),e)
    var_b = np.kron(XtX,sigma_ols)
    sigma_ml = (1/T)*np.matmul(np.transpose(e),e)
    A = np.linalg.cholesky(sigma_ols)

    return beta, A, sigma_ols

beta,A, sigma = VAR_OLS(data,1)



