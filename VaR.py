# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 15:06:29 2023

@author: jcrau
"""
import pandas as pd
import numpy as np

import yfinance as yf

from scipy import stats
import scipy.stats as stats
from scipy.stats import norm, shapiro, normaltest, anderson, kstest

import seaborn as sns
from matplotlib import pyplot as plt

from tabulate import tabulate

ticker = 'IBM'

accion = yf.Ticker(ticker)
hist = accion.history(period = '5Y')
precio = hist['Close'][:]
ret = np.zeros(len(precio))
for i in range(1,len(ret)):
    ret[i] = precio[i]/precio[i-1] - 1

sns.displot(ret, kde = True, bins = 30)
stats.probplot(ret, dist = "norm", plot =plt)

alpha = 0.05

def norm_test(alpha, df):
# Shapiro-Wilk, Kolmogorov-Smirnov and D'Agostino-Pearson Tests
    stat_SW, p_SW = shapiro(df)
    stat_KS, p_KS = kstest(df, "norm")
    stat_DP, p_DP = normaltest(df)

    SW = np.array([round(stat_SW,4), round(p_SW,2)])
    KS = np.array([round(stat_KS,4), round(p_KS,2)])
    DP = np.array([round(stat_DP,4), round(p_DP,2)]) 

    if p_SW > alpha:
        a = 'Sample looks Gaussian (fail to reject H0)'  
    else:
        a = 'Sample does not look Gaussian (reject H0)'  
    if p_KS > alpha:
        b = 'Sample looks Gaussian (fail to reject H0)'  
    else:
        b = 'Sample does not look Gaussian (reject H0)'     
    if p_DP > alpha:
        c = 'Sample looks Gaussian (fail to reject H0)'  
    else:
        c = 'Sample does not look Gaussian (reject H0)'

    SW = np.append(SW, a)
    KS = np.append(KS, b)
    DP = np.append(DP, c)

    col_names = ["Stat","p", "H0"]
    df=pd.DataFrame([SW, KS, DP],["Shapiro-Wilk","Kolmogorov-Smirnov", "D'Agostino-Pearson"],col_names)
    print(tabulate(df, headers = col_names, tablefmt = 'fancy_grid'))

norm_test(alpha, ret)
#%%
#Transformation

shape, loc, scale = stats.lognorm.fit(ret, loc=0)
pdf_lognorm = stats.lognorm.pdf(ret, shape, loc, scale)

sns.displot(pdf_lognorm, kde = True, bins = 30)
stats.probplot(pdf_lognorm, dist = "norm", plot =plt)

norm_test(alpha, pdf_lognorm)
#Box-Cox
pdf_lognorm_trans, lmbda = stats.boxcox(pdf_lognorm)

print('Best lambda parameter = %s' % round(lmbda, 3))

fig, ax = plt.subplots(figsize=(8, 4))
prob = stats.boxcox_normplot(pdf_lognorm, -20, 20, plot=ax)
ax.axvline(lmbda, color='r');

#%%        
def VaR1(n_shares, confidence_level, fecha):
            
    n_acciones = n_shares
    z = norm.ppf(1 - confidence_level)
    posicion = n_acciones * precio.loc[fecha]
    std = np.std(ret)
    VaR = posicion * z * std
    sns.displot(ret, kde = True, bins = 15)
    
    return print("Holdings=" ,posicion, "Var=",round(VaR,4), "tomorrow")

def VaR2(n_shares, confidence_level, n_dias, fecha):
        
    n_acciones = n_shares
    z = norm.ppf(confidence_level)
    posicion = n_acciones * precio.loc[fecha]
    std = np.std(ret)
    VaR = posicion * z * std * np.sqrt(n_dias)
    sns.displot(ret, kde = True, bins = 15)
    
    return print("Holdings=" ,posicion, "Var=",round(VaR,4), "en", n_dias, "Dias")

VaR1(1000, 0.99,'2018-06-06') 
VaR2(1000, 0.99, 10,'2018-06-06')