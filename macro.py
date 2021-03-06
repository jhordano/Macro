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

nber = np.array([ ['1953-07-01', '1954-05-01'],['1957-08-01', '1958-04-01'],
        ['1960-04-01', '1961-02-01'],['1969-12-01', '1970-11-01'],
        ['1973-11-01', '1975-03-01'],['1980-01-01', '1980-06-01'],
        ['1981-07-01', '1982-11-01'],['1990-07-01', '1991-03-01'],
        ['2001-03-01', '2001-11-01'],['2007-12-01', '2009-06-01']])

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

def graph_BBQ(serie,name,save_g):
    plt.figure()
    max_val = float(np.array(serie[name].max()))
    k = nber.shape[0]
    
    plt.plot(serie[name],color="darkblue")
    for w in range(k): 
        plt.fill_between(serie.index,0,max_val,where = (serie.index >= nber[w,0])&(serie.index <= nber[w,1]), 
                     facecolor='blue',alpha=0.2)
    
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().set_ylim(0, max_val)   
    
#***********  Peaks ***************      
    lines_P = np.array(serie[serie.BBQ_P == 1].index)
    for i in range(len(lines_P)):
        plt.axvline(x = str(lines_P[i]) ,color='r',linestyle="--",alpha=0.8)
    plt.show()
    plt.savefig(save_g + '_P.png')        
    
#***********  troughs ***************  
    plt.figure()
    plt.plot(serie[name],color="darkblue")
    for w in range(k): 
        plt.fill_between(serie.index,0,max_val,where = (serie.index >= nber[w,0])&(serie.index <= nber[w,1]), 
                     facecolor='blue',alpha=0.2)
    
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().set_ylim(0, max_val)   
    
    plt.plot(serie[name],color="darkblue")    
    lines_T = np.array(serie[serie.BBQ_T == 1].index)
    for i in range(len(lines_T)):
        plt.axvline(x = str(lines_T[i]) ,color='g',linestyle="--",alpha=0.8)
    plt.show()
    plt.savefig(save_g + '_T.png')
    
graph_BBQ(wage_adj,"wage_adj","wage_BBQ")

graph_BBQ(unemp,"LNS14000024","unemp_BBQ")

#graph_BBQ(gnp,"GNP")

#for i in range(2,10):
#    print(unemp.iloc[i,0])
# %%
for i in range(len(lines)):
    print(str(lines[i]))

# %% Seasonally Adjusted

# a good estimate of the seasonality cannot be made until the trend has been removed, and likewise a reliable estimate of the trend
# cannot be computed until the seasonality has been removed. 
 
# Dollars per Hour, Seasonally Adjusted
wage_sa = web.DataReader("CES3000000008", "fred", sdt,edt) 

# Dollars per Hour, NOT Seasonally Adjusted
wage = web.DataReader("AHEMAN", "fred", sdt, edt)
 
plt.figure()
plt.plot(wage_sa[wage_sa.index>"1990-01-01"],color="green",lw=1.8)
plt.plot(wage[wage.index>"1990-01-01"],color="yellow",lw=1.8)

# %%

##################
# First Stage
##################


T = len(wage)
wage['TD'] = np.nan
# Finding the trend
for i in range(6,T-6):                
          wage.iloc[i,1] = (wage.iloc[i-6,0] + 2*np.sum(wage.iloc[range(i-5,i+6),0]) + wage.iloc[i+6,0])/24    

wage['SI'] = wage['AHEMAN']/wage['TD']
wage['S'] = np.nan
wage['S_1'] = np.nan

# Finding S
for t in range(0,12):
    for i in range(6 + 12*2 + t, T -6 -12*2 , 12):                
          wage.iloc[i,3] = (wage.iloc[i-12*2,2]+2*wage.iloc[i-12,2]+3*wage.iloc[i,2]+2*wage.iloc[i+12,2]+wage.iloc[i+12*2,2])/9
          
# Finding S_1          
for i in range(6 + 12*2 + 6, T -6-12*2-6):                
          wage.iloc[i,4] = (wage.iloc[i-6,3] + 2*np.sum(wage.iloc[range(i-5,i+6),3]) + wage.iloc[i+6,3])/24    
wage['S'] = wage['S']/wage['S_1']
wage['SA'] = wage['AHEMAN']/wage['S']

##################
# Second Stage
##################
wage['SA_2'] = wage.SA.fillna(wage.AHEMAN)
# Finding the trend
for i in range(6,T-6):                
          wage.iloc[i,1] = (wage.iloc[i-6,6] + 2*np.sum(wage.iloc[range(i-5,i+6),6]) + wage.iloc[i+6,6])/24    
          
wage['SI'] = wage['AHEMAN']/wage['TD']

# Finding S
for t in range(0,12):
    for i in range(6 + 12*2 + t, T -6 -12*2 , 12):                
          wage.iloc[i,3] = (wage.iloc[i-12*2,2]+2*wage.iloc[i-12,2]+3*wage.iloc[i,2]+2*wage.iloc[i+12,2]+wage.iloc[i+12*2,2])/9
          
# Finding S_1          
for i in range(6 + 12*2 + 6, T -6-12*2-6):                
          wage.iloc[i,4] = (wage.iloc[i-6,3] + 2*np.sum(wage.iloc[range(i-5,i+6),3]) + wage.iloc[i+6,3])/24    
wage['S'] = wage['S']/wage['S_1']
wage['SA'] = wage['AHEMAN']/wage['S']


# %%

plt.figure()
plt.plot(wage_sa[(wage_sa.index>"1998-01-01")&(wage_sa.index<"2015-06-01")]["CES3000000008"],'bo',alpha=0.3)
plt.plot(wage[(wage.index>"1998-01-01")&(wage.index<"2015-06-01")]["SA"],color="red",lw=2)
plt.xlabel("Year")
plt.ylabel("Average hourly Earnings")
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["top"].set_visible(False)
plt.legend(['Published seasonally adjusted','Own procedure'])
plt.savefig("SA.png")
          
# %%          
for i in range(0,12):
    print(i)

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




