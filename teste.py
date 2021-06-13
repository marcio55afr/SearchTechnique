# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sktime.transformations.panel.dictionary_based import SAX, SFA, PAA
from sktime.utils.validation.panel import check_X
import scipy.stats
import matplotlib.pyplot as plt


from search_technique import SearchTechnique

#def extract():
df_train = pd.read_hdf('AtrialFibrillation', key='train')
labels = df_train.iloc[:,-1]

time_series =  pd.DataFrame([[ts.values] for ts in df_train.iloc[0:6,0]])
print(time_series)
st = SearchTechnique(640)
dfs = st.fit(time_series, labels[0:6])
#return dfs
    #fit.to_csv('df.csv')
    
def plot_Distribution():
    fit = pd.read_csv('df.csv')
    
    for i in range(15):
        rows = fit['sample']==i
    
        fit.loc[rows,'freq_relative'] = fit['frequency']/fit['total']
    
        fit.loc[rows,'freq_relative'].plot.hist(bins=100, ylim=(0,1000),xlim=(0,0.002))
        plt.figure()

    fit.to_csv('df_fr.csv')

def plot_ZipfsLaw():
    
    fit = pd.read_csv('df_fr.csv')
    
    
    fit = fit.sort_values(['sample','freq_relative'],ascending=False)
    

    for i in range(15):
        
        rows = fit['sample']==i
        n,_ = fit.loc[rows].shape
        fit.loc[rows,'rank'] = np.arange(1,n+1)
        f = fit.loc[rows,['rank','freq_relative']]
        
        plt.plot(f['rank'],f['freq_relative'])
        plt.xscale('log')
        plt.yscale('log')
    
    plt.show()
    
    plt.plot(f['rank'],f['freq_relative'],'r+')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
    
    fit.to_csv('df_rank.csv')

def test_all():
    
    #extract()
    plot_Distribution()
    plot_ZipfsLaw()

#dfs = extract()









