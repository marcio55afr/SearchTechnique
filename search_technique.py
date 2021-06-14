import math
import numpy as np
import pandas as pd
from sktime.classification.base import BaseClassifier

from sax_ngram import SaxNgram
from collections import Counter


__all__ = ["SearchTechnique"]


class SearchTechnique(BaseClassifier):
    
    '''
    Search Technique    
    
    
    Overview: Implementation of a search technique for feature selection in
    all the n-gram space.
    
    
    Parameters
    ----------
    
    initial_sample_size:    int, default = 2
                    size for the inicial sampling for each class to start
                    the searching, values between [...]
    
    random_state:   int, default = None
                    seed for the random functions

    '''


    def __init__(self,
                 max_series_length,
                 min_window_length = 6,
                 window_prop = .5,
                 dimension_reduction_prop=.80,
                 alphabet_size=4,
                 initial_sample_per_class=2,
                 random_state=None):
        
        self.max_series_length = max_series_length
        self.min_window_length = min_window_length
        self.window_prop = window_prop
        self.dimension_reduction_prop = dimension_reduction_prop
        self.alphabet_size = alphabet_size
        self.initial_sample_per_class = initial_sample_per_class
        self.random_state = random_state
        
        # Variables
        self._transformer = SaxNgram(self.max_series_length,
                                     self.min_window_length,
                                     self.window_prop,
                                     self.dimension_reduction_prop,
                                     self.alphabet_size)
        
    def fit(self, data, labels):
        
        data_length,_ = data.shape
        # TODO check number of samples and labels
        classes = labels.unique()
        sample_size = self.initial_sample_per_class
        # TODO other types of selection
        # random selection over all data
        while(True):

            # TODO unbalanced classes
            samples = self._get_samples(sample_size, classes, labels)
            N = samples.size
            print('\nTransforming the series..')
            print('\nsample size: {}'.format(N))
            
            # TODO make the transformation multivariable!!!! important
            bags = self._transformer.fit_transform(data.iloc[samples,:])
            dfs = pd.DataFrame([])
            
            print('\nCalculating the tf of each word..\n')
            i=0
            for sample in samples:
                # TODO separate the words from the meta informations
                # for speed up
                df = pd.DataFrame.from_dict(bags[i], orient='index')
                df = df.reset_index()
                df.columns = ['word', 'frequency']
                df['resolution'] = [ ' '.join(word.split(' ')[0:2]) for word in df['word']]
                df['sample'] = sample
                df['total'] = df.shape[0]
                df['tf'] = df['frequency']/df['total']
                df.index.names = ['index']
                df = df.reset_index().set_index(['sample','index'])
                
                dfs = pd.concat([dfs,df], axis=0, join='outer')
                
                i+=1
            
            print('\nCalculating the tf idf of each word..')
            # TODO remove from N the 'documents' from the same class
            dfs['tf_idf'] = 0
            docs_freqs = pd.Series(Counter(dfs['word']))
            i=0
            i_max = samples.size
            for sample in samples:
                i+=1
                if(i > (i_max/10)):
                    print('#',end=' ')
                    i -= i_max//10
                
                df = dfs.loc[sample].sort_values('word')
                tf = df['tf']
                words = df['word']
                
                doc_freq = docs_freqs[words]
                idf = np.log2(N/(doc_freq+1))
                
                dfs.loc[sample,'tf_idf'] = tf*idf

            if( samples.size == labels.size ):
                print('\n\nAll samples were processed!')
                break
            
            # TODO test this parameter of half parameters double samples
            # with other proportions like 1/3 3* or sum with fixed buckets
            sample_size *= 2

            dfs = dfs.reset_index()
            dfs = dfs.sort_values('tf_idf')

            print('\nRemoving worst resolutions found in this interation')
            # TODO compare half of parameters against half of the words
            last_resolutions = dfs[['resolution','tf_idf']].groupby('resolution').max()
            last_resolutions = last_resolutions.sort_values('tf_idf').index
            print('last set of resolutions:')
            print(last_resolutions)
            
            last_worst_resolutions = self._half_split(last_resolutions)
            if(last_worst_resolutions.size == dfs['resolution'].unique().size):
                raise 'The worst resolutions comprehense all the resolution \
                        isntead of half of resolutions'
            
            # TODO Vote system to down vote
            # necessarily to up vote in other versions of ST
            self._transformer.remove_resolutions(last_worst_resolutions)

        return dfs
        
     
        
    def predict_proba(self):
        return .0
    
    def predict(self):
        return 0
    
    def _get_samples(self, sample_size, classes, labels):
        
        samples = pd.Series([])
        for c in classes:
            # TODO random State, replication of experiments
            index = pd.Series(labels[labels==c].index)
            if(sample_size > index.size):
                samples = samples.append(index)
                continue
            
            s = index.sample(sample_size)
            samples = samples.append(s)
        
        return samples
        
    def _get_histogram(self):
        return pd.DataFrame([])
    
    def _half_split(self, data, part='first'):
        
        half = len(data)//2
        
        if(part=='first'):
            return data[:half]
        
        elif(part=='second'):
            return data[half:]

        raise 'The function _half_split can only return the first or second part'



























