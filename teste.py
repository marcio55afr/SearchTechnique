# -*- coding: utf-8 -*-

import os.path
import numpy as np
import pandas as pd
from sktime.transformations.panel.dictionary_based import SAX, SFA, PAA
from sktime.utils.validation.panel import check_X
from sktime.benchmarking.data import UEADataset
from sklearn.metrics import accuracy_score
import scipy.stats
import matplotlib.pyplot as plt

from search_technique import SearchTechnique


class Test(object):
    

    def __init__(self, name, path, dataset, ts_length, initial_sample=2):
        
        self.name = name
        self.THRESHOLD = [0,1,2,3]
        self.PATH = path
        self.dataset = dataset
        self.ts_length = ts_length
        
        CSV_PATH = self.PATH + 'csv/' + self.dataset
        self.EXTRACT_PATH = CSV_PATH+'extract_threshold_'
        self.DISTRIBUTION_PATH = CSV_PATH+'distribution_threshold_'
        self.RANK_PATH = CSV_PATH+'rank_threshold_'
        
        self.initial_sample = initial_sample    
    
    def extract(self, threshold):
        df_train = pd.read_hdf(self.dataset, key='train')
        labels = df_train.iloc[:,-1]
        
        time_series =  pd.DataFrame([[ts.values] for ts in df_train.iloc[:,0]])
        print(time_series)
        st = SearchTechnique(self.ts_length,initial_sample_per_class=self.initial_sample, random_state = 12)
        st._transformer._frequency_thereshold = threshold
        dfs = st.fit(time_series, labels)
        dfs = dfs.rename(columns={'sample_id':'sample'})
        #return dfs
        dfs.to_csv(self.EXTRACT_PATH + str(threshold) + '.csv')
    
    def write_distribution(self, threshold):
        fit = pd.read_csv(self.EXTRACT_PATH + str(threshold) + '.csv')
        
        for sample_id in fit['sample'].unique():
            rows = fit['sample']== sample_id
            fit.loc[rows,'total'] = rows.sum()
        fit['freq_relative'] = fit['frequency']/fit['total']
            
        fit.to_csv(self.DISTRIBUTION_PATH + str(threshold) + '.csv')
        
    def plot_distribution(self, threshold):
        fit = pd.read_csv(self.DISTRIBUTION_PATH + str(threshold) + '.csv')

        fig, axs = plt.subplots(5,3,
                                sharex=True,
                                sharey=True,
                                figsize=(10,8),
                                dpi=150)
        indices = [(x,y) for x in range(5) for y in range(3)]
        for sample_id, (x, y) in zip(fit['sample'].unique(), indices):
            ax = axs[x,y]
            rows = fit['sample']== sample_id
            ax.hist(fit.loc[rows,'freq_relative'],bins=200)
            #ax.set_xlim(0,0.004)
            #ax.set_ylim(0,500)
            ax.set_title('sample {}'.format(sample_id), )
        fig.supxlabel('word relative frequency')
        fig.supylabel('Frequency')
        fig.suptitle('Frequency distribution with threshold {}'.format(threshold))
        plt.tight_layout(rect = (0.02,0,1,0.98))
        plt.show()
    
    
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
        
        #plt.style.use('seaborn-paper')
        plt.figure(figsize=(8,6), dpi=150)
        for sample_id in fit['sample'].unique():
            
            rows = fit['sample']== sample_id
            f = fit.loc[rows,['rank','frequency','freq_relative']]
            
            plt.plot(f['rank'],f['freq_relative'], label='sample '+str(sample_id))
            plt.xscale('log')
            plt.yscale('log')
            #plt.xlim(0.6,100000)
            #ylim0, ylim1 = plt.ylim()
            #ylim1 = 0.15
            #plt.ylim(0.00009, ylim1)
        
        #plt.legend(title='"Documents"',
        #           loc='upper right',
        #           ncol=2,
        #           fontsize='x-small')
        plt.title('{} test and threshold {}'.format(self.name,threshold))
        plt.xlabel('word rank')
        plt.ylabel('word frequency')
        plt.show()

        
    def plot_zipfsLaw(self, threshold):
        
        fit = pd.read_csv(self.RANK_PATH + str(threshold) + '.csv')
        
        sample_id = fit['sample'].sample(1).iloc[0]
        f = fit.loc[fit['sample']==sample_id,['rank','frequency','freq_relative']]       
        
        #plt.style.use('seaborn-paper')
        plt.figure(figsize=(8,6), dpi=150)
        plt.plot(f['rank'],f['freq_relative'],'r+', label='sample '+str(sample_id))
        plt.xscale('log')
        plt.yscale('log')
        #plt.xlim(0.6,100000)
        #ylim0, ylim1 = plt.ylim()
        #ylim1 = 0.15
        #plt.ylim(0.00009, ylim1)
        
        #plt.legend(title='"Document"', 
        #           loc='upper right',
        #           ncol=2,
        #           fontsize='x-small')
        plt.title('Zipf\'s Law - {} test and threshold {}'.format(self.name,threshold))
        plt.xlabel('Word rank')
        plt.ylabel('Word Frequency')
        plt.show()
        

    def run_test(self):
        
        for threshold in self.THRESHOLD:
            if(not os.path.isfile(self.EXTRACT_PATH + str(threshold) + '.csv')):
                self.extract(threshold)
                self.write_distribution(threshold)
                self.write_all_zipfslaw(threshold)

        for threshold in self.THRESHOLD: self.plot_distribution(threshold)
        for threshold in self.THRESHOLD: self.plot_all_zipfslaw(threshold)
        for threshold in self.THRESHOLD: self.plot_zipfsLaw(threshold)


resolutions = {
     4: [1,2,3,4,5,6,7],
     7: [1,2,3,4],
    10: [1,2,3],
    14: [1,2],
    16: [1],
    19: [1],
    22: [1],
    25: [1],
    28: [1]
    }

resolution_matrix = pd.DataFrame( False, columns=resolutions.keys(), index=np.arange(7)+1 )
test_word_matrix = pd.DataFrame( False, columns=resolutions.keys(), index=np.arange(7)+1 )

for window, ngrams in resolutions.items():
    for n in ngrams:
        DATA_PATH = os.path.join(os.path.abspath(os.getcwd()), "experiments/datasets/univariate/")
        df = UEADataset(path=DATA_PATH, name='SyntheticControl').load()
        df['index'] = range(df.shape[0])
        
        
        df_train = df.loc['train'].set_index('index')
        labels = df_train.iloc[:,-1]
        
        time_series =  pd.DataFrame([[ts.values] for ts in df_train.iloc[:,0]])
        #print(time_series)
        st = SearchTechnique(random_state=31)
        st = st.fit(time_series, labels, n, window)
        
        print('\n\n\n Testing the dataset')
        
        df_test = df.loc['test'].set_index('index')
        labels = df_test.iloc[:,-1]
        time_series =  pd.DataFrame([[ts.values] for ts in df_test.iloc[:,0]])
        time_series.index = df_test.index
        #print(time_series)
        labels_pred = st.predict(time_series)
        
        test_word_matrix.loc[n,window] = st._num_test_words
        
        print(st._num_test_words)
        
        acc = accuracy_score(labels.values, labels_pred)
        print(window, n, acc)
        resolution_matrix.loc[n, window] = acc

        
        
'''
DATA_PATH = os.path.join(os.path.abspath(os.getcwd()), "experiments/datasets/univariate/")
df = UEADataset(path=DATA_PATH, name='SyntheticControl').load()
df['index'] = range(df.shape[0])


df_train = df.loc['train'].set_index('index')
labels = df_train.iloc[:,-1]

time_series =  pd.DataFrame([[ts.values] for ts in df_train.iloc[:,0]])
#print(time_series)
st = SearchTechnique(random_state=31)
word_summary = st.fit(time_series, labels)
word_summary.to_csv('experiments/tf-idf_significance/word_summary.csv')
'''  
        

'''

path = 'results/initial_test/'
test = Test('fisrt',path, 'JapaneseVowels', 29)
test.run_test()

path = 'results/initial_test/'
test = Test('fisrt',path, 'AtrialFibrillation', 640)
test.run_test()

'''
'''
path = 'results/initial_test/second_test/'
test = Test('second',path,2)
test.THRESHOLD = [2]
test.run_test()
'''


















