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

mdata.index = pandas.DatetimeIndex(quarterly)

data = np.log(mdata).diff().dropna()

 # make a VAR model
model = VAR(data)

results = model.fit(2)
print(results.summary())

results.plot()
# Plotting time series autocorrelation function:
results.plot_acorr()

# one can pass a maximum number of lags and the order criterion to use for order selection:
results = model.fit(maxlags=15, ic='aic')

irf = results.irf(10)
irf.plot(orth=False)


# %%

lags = 2
T,k = data.shape

Y = data.iloc[range(k-1,T)]
X = pd.DataFrame([])
for i in range(1,k):
    print(i)
    X_k = data.iloc[range(k-i,T-i)]
    X_k.rename(columns=lambda x: x +'_' +str(i), inplace=True)
    X = pd.concat([X,X_k],axis=1)
    