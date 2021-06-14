# -*- coding: utf-8 -*-

import os.path
import numpy as np
import pandas as pd
from sktime.transformations.panel.dictionary_based import SAX, SFA, PAA
from sktime.utils.validation.panel import check_X
import scipy.stats
import matplotlib.pyplot as plt

from search_technique import SearchTechnique


class Test(object):

    def __init__(self, name, path, initial_sample):
        
        self.name = name
        self.THRESHOLD = [0,1,2,3]
        self.PATH = path
        CSV_PATH = self.PATH+'csv/'
    
        self.EXTRACT_PATH = CSV_PATH+'extract_threshold_'
        self.DISTRIBUTION_PATH = CSV_PATH+'distribution_threshold_'
        self.RANK_PATH = CSV_PATH+'rank_threshold_'
        
        self.initial_sample = initial_sample
    
    
    def extract(self, threshold):
        df_train = pd.read_hdf('AtrialFibrillation', key='train')
        labels = df_train.iloc[:,-1]
        
        time_series =  pd.DataFrame([[ts.values] for ts in df_train.iloc[:,0]])
        print(time_series)
        st = SearchTechnique(640,initial_sample_per_class=self.initial_sample)
        st._transformer._frequency_thereshold = threshold
        dfs = st.fit(time_series, labels)
        #return dfs
        dfs.to_csv(self.EXTRACT_PATH + str(threshold) + '.csv')
    
    def write_distribution(self, threshold):
        fit = pd.read_csv(self.EXTRACT_PATH + str(threshold) + '.csv')
        
        for sample_id in fit['sample'].unique():
            rows = fit['sample']== sample_id
            fit.loc[rows,'freq_relative'] = fit['frequency']/fit['total']
    
        fit.to_csv(self.DISTRIBUTION_PATH + str(threshold) + '.csv')
        
    def plot_distribution(self, threshold):
        fit = pd.read_csv(self.DISTRIBUTION_PATH + str(threshold) + '.csv')
        
        for sample_id in fit['sample'].unique():
            rows = fit['sample']== sample_id        
            fit.loc[rows,'freq_relative'].plot.hist(bins=100)#, ylim=(0,2000),xlim=(0,0.005))
            plt.figure()
    
    
    def write_all_zipfslaw(self, threshold):
        fit = pd.read_csv(self.DISTRIBUTION_PATH + str(threshold) + '.csv')
        fit = fit.sort_values(['sample','freq_relative'],ascending=False)
    
        for sample_id in fit['sample'].unique():
    
            rows = fit['sample']== sample_id
            n,_ = fit.loc[rows].shape
            fit.loc[rows,'rank'] = np.arange(1,n+1)
            
        fit.to_csv(self.RANK_PATH + str(threshold) + '.csv')
            
    def plot_all_zipfslaw(self, threshold):
        fit = pd.read_csv(self.RANK_PATH + str(threshold) + '.csv')
        fit = fit.sort_values(['sample','freq_relative'],ascending=False)
        
        plt.figure(figsize=(8,6), dpi=150)
        for sample_id in fit['sample'].unique():
            
            rows = fit['sample']== sample_id
            f = fit.loc[rows,['rank','frequency','freq_relative']]
            
            plt.plot(f['rank'],f['frequency'], label='sample '+str(sample_id))
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim(0.6,100000)
        
        plt.legend(title='"Documents"',
                   loc='upper right',
                   ncol=2,
                   fontsize='x-small')
        plt.title('{} test and threshold {}'.format(self.name,threshold))
        plt.xlabel('Word rank')
        plt.ylabel('Word Frequency')
        plt.show()

        
    def plot_zipfsLaw(self, threshold):
        
        fit = pd.read_csv(self.RANK_PATH + str(threshold) + '.csv')
        
        sample_id = fit['sample'].sample(1).iloc[0]
        f = fit.loc[fit['sample']==sample_id,['rank','frequency','freq_relative']]       
        
        plt.figure(figsize=(8,6), dpi=150)
        plt.plot(f['rank'],f['frequency'],'r+', label='sample '+str(sample_id))
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim(0.6,100000)
        
        plt.legend(title='"Document"', 
                   loc='upper right',
                   ncol=2,
                   fontsize='x-small')
        plt.title('Zipf\'s Law - {} test and threshold {}'.format(self.name,threshold))
        plt.xlabel('Word rank')
        plt.ylabel('Word Frequency')
        plt.show()
        

    def run_test(self):
        
        for threshold in self.THRESHOLD:
            if(not os.path.isfile(self.EXTRACT_PATH + str(threshold) + '.csv')):
               self.extract(threshold)

        #for threshold in self.THRESHOLD: self.plot_Distribution(threshold)
        for threshold in self.THRESHOLD: self.plot_all_zipfslaw(threshold)
        for threshold in self.THRESHOLD: self.plot_zipfsLaw(threshold)

    
path = 'results/initial_test/'
test = Test('fisrt',path,15)
test.run_test()

path = 'results/second_test/'
test = Test('second',path,2)
test.THRESHOLD = [3]
test.run_test()









