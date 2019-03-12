# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 16:21:32 2019

@author: s3179575
"""
# %reset
# http://docentes.fe.unl.pt/~frafra/Site/course_Empirical_Macro_spring_2016.html

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
from numpy.linalg import inv

os.chdir("X:\My Documents\Share_g\Macro")
data_F = pd.read_excel('FREDQD_T.xlsx', index_col=0) 
y = data_F.iloc[:,0]
X_B_D = data_F.iloc[:,1:]
X_B = np.array(data_F.iloc[:,1:])
T, p = X_B.shape
XTX = (1/T)*np.matmul(np.transpose(X_B),X_B)

D,V = LA.eig(XTX)
F = np.matmul(X_B,V)


#%% 
plt.figure()
plt.plot(F[:,0])
plt.show()

# %%
# Question 1
k = np.min([T,p])
de = np.sort(D)[::-1]
Exp_V = np.zeros(k)
TT = np.sum(de[:k])

for ll in range(k):
    Exp_V[ll] = de[ll]/TT    

plt.figure()
plt.plot(Exp_V)
plt.show()

# %%
# Question 2
Load = np.transpose(V)
for i in range(3):
    plt.figure()
    plt.plot(Load[i])
    plt.show()


# %%
# Question 3
lags = 4
l_k = 2

def factor_model(y, X_B_D, lags,l_k):    
    X_B = np.array(X_B_D)
    T, p = X_B.shape

    Y = y.iloc[range(lags,T)]  
#======= Lags of the endogenous variable ===================
    Y_r = pd.DataFrame(np.ones(T-lags),columns=['c'])    
    for i in range(lags,0,-1):    
            Y_re = y.iloc[range(i-1,T-1-(lags-i))].reset_index(drop=True)
            Y_re.rename(columns=lambda x: x +'_' +str(lags+1-i), inplace=True)
            Y_r = pd.concat([Y_r,Y_re],axis=1)        
#======= Factors ===================            
    XTX = (1/T)*np.matmul(np.transpose(X_B),X_B)
    D,V = LA.eig(XTX)
    F = np.matmul(X_B,V)
    F_r = F[range(lags-1,T-1),:l_k].real   
       
    Y = np.matrix(Y).reshape((-1,1))
    Y_r_X =np.concatenate((np.matrix(Y_r),F_r), axis=1)
    YrXtYrX = np.matmul(np.transpose(Y_r_X),Y_r_X)
    beta = np.matmul(inv(YrXtYrX),np.matmul(np.transpose(Y_r_X),Y))
    T = T-lags
    k = lags + l_k
    e = Y - np.matmul(Y_r_X,beta)
    sigma_ml = (1/T)*np.matmul(np.transpose(e),e)
    V_p = np.log(sigma_ml)
    AIC = V_p + (2/T)*(k+1)
    BIC = V_p + (np.log(T)/T)*(k+1)
    
#    y_r_p = np.transpose(np.append(np.ones((1,1)),Y[-lags:,0][::-1],axis=0))
#    y_pr = np.matmul(y_r_p,beta)[0,0] 
    return beta, AIC, BIC

beta, AIC, BIC = factor_model(y,X_B_D,lags,l_k)

# %%
AIC_S = np.zeros(12)
BIC_S = np.zeros(12)
for ll in range(1,13):
    beta, AIC, BIC = factor_model(y,X_B_D,lags,ll)
    AIC_S[ll-1] = AIC
    BIC_S[ll-1] = BIC

plt.figure()
plt.plot(AIC_S)
plt.plot(BIC_S)
plt.show()


# %%
# Question 4
lags = 4
l_k = 1

def proyec_ar(y, lags):
    T = y.shape[0]
    Y = y.iloc[range(lags,T)]   
    Y_r = pd.DataFrame(np.ones(T-lags),columns=['c'])
        
    for i in range(lags,0,-1):    
            Y_re = y.iloc[range(i-1,T-1-(lags-i))].reset_index(drop=True)
            Y_re.rename(columns=lambda x: x +'_' +str(lags+1-i), inplace=True)
            Y_r = pd.concat([Y_r,Y_re],axis=1)
       
    Y = np.matrix(Y).reshape((-1,1))
    Y_r = np.matrix(Y_r)
    YrtYr = np.matmul(np.transpose(Y_r),Y_r)
    beta = np.matmul(inv(YrtYr),np.matmul(np.transpose(Y_r),Y))
    y_r_p = np.transpose(np.append(np.ones((1,1)),Y[-lags:,0][::-1],axis=0))
    y_pr = np.matmul(y_r_p,beta)[0,0] 
    return y_pr

dates = y.index
start = y.index.get_loc('2003-10-01')
T_p = len(y.index)

y_pro = np.zeros((T_p-start))
for i in range(start,T_p):
    y_data = y[y.index < dates[i]]
    y_pro[i-start] = proyec_ar(y_data,lags)

y_pro = pd.DataFrame(y_pro)
y_pro.index = dates[start:]


# %%
# 1959Q4 and 2003Q3.
y_real = y[y.index > '2003-07-01']
plt.figure()
plt.plot(y_real,lw=2,color="darkblue")
plt.plot(y_pro,lw=1.5,color="green", ls='--')
plt.show()    



 




