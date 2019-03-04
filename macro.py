# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 14:09:25 2019

@author: HP PROBOOK
"""



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
wage_adj = wage_adj.rename(columns={"CES3000000008":"wage_adj"})
wage_adj['wage_adj'] = np.array((wage_adj['wage_adj'])/np.array(IPC['GNPDEF'])*100)
 
# Average Hourly Earnings of Production and Nonsupervisory Employees: Manufacturing
# Dollars per Hour, NOT Seasonally Adjusted
wage = web.DataReader("CEU0500000008", "fred", sdt, edt) 
wage.head()

# %%


# Gros national product
gnp = web.DataReader("GNP","fred",sdt,edt)

# Unemployment Rate: 20 years and over (LNS14000024)
# Units: Percent, Seasonally Adjusted
unemp = web.DataReader("LNS14000024","fred",sdt,edt)
unemp = unemp.groupby(pd.PeriodIndex(unemp.index, freq='Q')).mean()


# P L O T S 
#from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from tabulate import tabulate

def graph_a(serie,text):
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
    plt.gca().set_ylim(0, max_val)
    #plt.title("Average hourly earnings")
    #plt.text('1970-01-01', 0, "comic sans", family="Comic Sans MS")
    plt.show()

graph_a(wage_adj,"Average hourly Earnings")
graph_a(gnp,"Gros National Product")
graph_a(unemp,"Unemployment Rate")


from tabulate import tabulate

def table_a(serie):
    mean = np.array(serie.mean())
    std = np.array(serie.std())
    q1 = np.array(serie.quantile(0.8))
    H1 = np.array(["stat","1950-1990"])
    C1 = np.array(["mean", "std dev", "Q4"]).reshape(3,1)        
    coef = np.array([mean,std,q1]).reshape(3,1)        
    d_p = np.concatenate((C1,coef),axis=1).reshape(3,2)
    table = tabulate(d_p , headers = H1, floatfmt=".4f") 
    return table

table_a(gnp)

## Question b
# BBQ Rule
# %%

# Peak: y_t+k ; : : : ; y_t-1 < y_t > y_t+1; : : : ; yt+k
def bbq(serie):
    serie['bbq_P'] =0 
    serie['bbq_T'] =0 
    
    T = serie.shape[0]
    k = 2 
    for i in range(k,T-k):
        if (serie.iloc[i,0] > np.max(serie.iloc[(i-k):i,0]))&(serie.iloc[i,0] > np.max(serie.iloc[(i+1):(i+3),0])) : 
            serie.iloc[i,1] = 1 
    
        if (serie.iloc[i,0] < np.min(serie.iloc[(i-k):i,0]))&(serie.iloc[i,0] < np.min(serie.iloc[(i+1):(i+3),0])) : 
            serie.iloc[i,2] = 1 
            
bbq(unemp)        
bbq(wage_adj)
bbq(gnp)        
#for i in range(2,10):
#    print(unemp.iloc[i,0])





