Created on Mon Jun  5 15:06:29 2023

@author: jcrau
"""
# Importar paquetes
import pandas as pd
import numpy as np

import yfinance as yf

from scipy import stats
import scipy.stats as stats
from scipy.stats import norm, shapiro, normaltest, anderson, kstest

import seaborn as sns
from matplotlib import pyplot as plt

from tabulate import tabulate
# Importar información de mercado
ticker = 'IBM'
accion = yf.Ticker(ticker)
hist = accion.history(period = '5Y')
#Calcular retornos
precio = hist['Close'][:]
ret = np.zeros(len(precio))
for i in range(1,len(ret)):
    ret[i] = precio[i]/precio[i-1] - 1
# Análisis gráfico de la distribución y prueba Q-Q
sns.displot(ret, kde = True, bins = 30)
stats.probplot(ret, dist = "norm", plot =plt)
#Definir pruebas de normalidad
def norm_test(alpha, df):
# Tests de Shapiro-Wilk, Kolmogorov-Smirnov and D'Agostino-Pearson
    stat_SW, p_SW = shapiro(df)
    stat_KS, p_KS = kstest(df, "norm")
    stat_DP, p_DP = normaltest(df)

    SW = np.array([round(stat_SW,4), round(p_SW,2)])
    KS = np.array([round(stat_KS,4), round(p_KS,2)])
    DP = np.array([round(stat_DP,4), round(p_DP,2)]) 

    if p_SW > alpha:
        a = 'La muestra se ve Gaussiana (No rechazar H0)'  
    else:
        a = 'La muestra no se ve Gaussiana (rechazar H0)'  
    if p_KS > alpha:
        b = 'La muestra se ve Gaussiana (No rechazar H0)'  
    else:
        b = 'La muestra no se ve Gaussiana (rechazar H0)'     
    if p_DP > alpha:
        c = 'La muestra se ve Gaussiana (No rechazar H0)'  
    else:
        c = 'La muestra no se ve Gaussiana (rechazar H0)'

    SW = np.append(SW, a)
    KS = np.append(KS, b)
    DP = np.append(DP, c)

    col_names = ["Stat","p", "H0"]
    df=pd.DataFrame([SW, KS, DP],["Shapiro-Wilk","Kolmogorov-Smirnov", "D'Agostino-Pearson"],col_names)
    print(tabulate(df, headers = col_names, tablefmt = 'fancy_grid'))

alpha = 0.05
norm_test(alpha, ret)

# Transformaciones
# En caso la data 
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
