# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:09:25 2019

@author: HP PROBOOK
"""
import os
os.chdir("X:\My Documents\Share_g\Macro")

import pandas as pd 
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web 
import numpy as np 
import datetime as dt

# SET START AND END DATES OF THE SERIES 
sdt = dt.datetime(1948, 1, 1) 
edt = dt.datetime(2018, 9, 1) 

#  Gross National Product: Implicit Price Deflator
IPC = web.DataReader("GNPDEF","fred",sdt,edt)

# Average Hourly Earnings of Production and Nonsupervisory Employees: Manufacturing
# Dollars per Hour, Seasonally Adjusted
wage_adj = web.DataReader("CES3000000008", "fred", sdt,edt) 

wage_adj = wage_adj.groupby(pd.PeriodIndex(wage_adj.index, freq='Q')).mean()
wage_adj.index = wage_adj.index.to_timestamp()
wage_adj = wage_adj.rename(columns={"CES3000000008":"wage_adj"})
wage_adj['wage_adj'] = ((np.array(wage_adj['wage_adj'])/np.array(IPC['GNPDEF']))*100)
 

# Average Hourly Earnings of Production and Nonsupervisory Employees: Manufacturing
# Dollars per Hour, NOT Seasonally Adjusted
wage = web.DataReader("CEU0500000008", "fred", sdt, edt) 
wage = wage.groupby(pd.PeriodIndex(wage.index, freq='Q')).mean()
wage.index = wage.index.to_timestamp()
wage.head()

# %%


# Gros national product
gnp = web.DataReader("GNP","fred",sdt,edt)

# Unemployment Rate: 20 years and over (LNS14000024)
# Units: Percent, Seasonally Adjusted
unemp = web.DataReader("LNS14000024","fred",sdt,edt)
unemp = unemp.groupby(pd.PeriodIndex(unemp.index, freq='Q')).mean()
unemp.index = unemp.index.to_timestamp()

# P L O T S 
#from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from tabulate import tabulate

def graph_a(serie,text,save_g):
    plt.figure()
    max_val =np.array(serie.max())    
    plt.plot(serie,lw=2,color="blue")
    plt.fill_between(serie.index,0,max_val[0],where = serie.index >= '1991-01-01', 
                     facecolor='blue',alpha=0.2)
    plt.xlabel("Year")
    plt.ylabel(text)
    # Hide the right and top spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().set_ylim(0, max_val*1.1)
    #plt.title("Average hourly earnings")
    #plt.text('1970-01-01', 0, "comic sans", family="Comic Sans MS")
    plt.show()
    plt.savefig(save_g + '.png') # produce PNG

graph_a(wage_adj,"Average hourly Earnings",'wage')

graph_a(gnp,"Gros National Product",'gnp')
graph_a(unemp,"Unemployment Rate",'unemp')

# %%
from tabulate import tabulate

def table_a(serie):
    H1 = np.array(["stat","1948-1990","1991-2018","Total" ])
    C1 = np.array(["mean", "std dev", "Q4"]).reshape(3,1)
    # "1948-1990"
    mean = np.array(serie[serie.index < "1991-01-01"].mean())
    std = np.array(serie[serie.index < "1991-01-01"].std())
    q1 = np.array(serie[serie.index < "1991-01-01"].quantile(0.8))
    coef1 = np.array([mean,std,q1]).reshape(3,1) 
    
    # "1991-2018"
    mean_2 = np.array(serie[serie.index >= "1991-01-01"].mean())
    std_2 = np.array(serie[serie.index >= "1991-01-01"].std())
    q1_2 = np.array(serie[serie.index >= "1991-01-01"].quantile(0.8))
    coef2 = np.array([mean_2,std_2,q1_2]).reshape(3,1)

    # "1948-2018"
    mean_T = np.array(serie.mean())
    std_T = np.array(serie.std())
    q1_T = np.array(serie.quantile(0.8))
    coefT = np.array([mean_T,std_T,q1_T]).reshape(3,1)
       
    d_p = np.concatenate((C1,coef1,coef2,coefT),axis=1).reshape(3,4)
    table = tabulate(d_p , headers = H1, floatfmt=".2f") 
    return table 

print(table_a(gnp["GNP"]))
print(table_a(unemp["LNS14000024"]))
print(table_a(wage_adj["wage_adj"]))


# %%
## Question b
# BBQ Rule

# Peak: y_t+k ; : : : ; y_t-1 < y_t > y_t+1; : : : ; yt+k
def BBQ(serie):
    serie['BBQ_P'] =0 
    serie['BBQ_T'] =0 
    
    T = serie.shape[0]
    k = 2 
    for i in range(k,T-k):
        if (serie.iloc[i,0] > np.max(serie.iloc[(i-k):i,0]))&(serie.iloc[i,0] > np.max(serie.iloc[(i+1):(i+3),0])) : 
            serie.iloc[i,1] = 1 
    
        if (serie.iloc[i,0] < np.min(serie.iloc[(i-k):i,0]))&(serie.iloc[i,0] < np.min(serie.iloc[(i+1):(i+3),0])) : 
            serie.iloc[i,2] = 1 
            
BBQ(unemp)        
BBQ(wage_adj)
BBQ(gnp)        
#for i in range(2,10):
#    print(unemp.iloc[i,0])

# %% Seasonally Adjusted

# Dollars per Hour, Seasonally Adjusted
wage_sa = web.DataReader("CES3000000008", "fred", sdt,edt) 
# Dollars per Hour, NOT Seasonally Adjusted
wage = web.DataReader("CEU0500000008", "fred", sdt, edt) 

plt.figure()
plt.plot(wage_sa)
plt.plot(wage)

# %%
gnp['SA'] = gnp["GNP"]

T = len(gnp)
for i in range(6,T-6):                
          gnp.iloc[i,3] = (gnp.iloc[i-6,0] + 2*np.sum(gnp.iloc[range(i-5,i+6),0]) + gnp.iloc[i+6,0])/24    



# %%
gnp.GNP.plot()
gnp.SA.plot()




# %%
#  Unit root test


from statsmodels.tsa.stattools import adfuller
result = adfuller(np.log(gnp.GNP))
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

from arch.unitroot import ADF
adf = ADF(np.log(gnp.GNP))
print('ADF Statistic: {0:0.4f}'.format(adf.stat))
print('p-value: {0:0.4f}'.format(adf.pvalue))

from arch.unitroot import DFGLS
dfgls = DFGLS(np.log(gnp.GNP))
print('DFGLS Statistic: {0:0.4f}'.format(dfgls.stat))
print('p-value: {0:0.4f}'.format(dfgls.pvalue))


from arch.unitroot import PhillipsPerron
pp = PhillipsPerron(np.log(gnp.GNP))  
print('Phillips Perron Statistic: {0:0.4f}'.format(pp.stat))
print('p-value: {0:0.4f}'.format(pp.pvalue))


from arch.unitroot import KPSS
kps = PhillipsPerron(np.log(gnp.GNP))  
print('KPSS  Statistic: {0:0.4f}'.format(kps.stat))
print('p-value: {0:0.4f}'.format(kps.pvalue))




