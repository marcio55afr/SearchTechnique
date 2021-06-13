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
        # TODO check class column
        classes = labels.unique()
        n_classes = classes.size
        sample_size = self.initial_sample_per_class
        # TODO other types of selection
        # random selection over all data
        _finished = False
        while(not _finished):

            # TODO unbalanced classes
            samples = self._get_samples(sample_size, classes, labels)
            N = samples.size
            print('\n\n sample size: {}'.format(N))
            
            bags = self._transformer.fit_transform(data.iloc[samples,:])
            dfs = pd.DataFrame([])
            
            i=0
            for sample in samples:
                # TODO separate the words from the meta informations
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
            
            # TODO remove from N the 'documents' from the same class
            dfs['tf_idf'] = 0
            df = Counter(dfs['word'])
            for sample in samples:
                i = 0
                while True:
                    try:
                        row = dfs.loc[(sample, i)]
                    except:
                        break
                    word = row['word']                    
                    tf = row['tf']
                    idf = np.log2(N/(df[word]+1))
                    dfs.loc[(sample, i),'tf_idf'] = tf*idf
                    i+=1

            if( samples.size == labels.size ):
                _finished = True
            
            # TODO test this parameter of half parameters double samples
            sample_size *= 2
                        
            # TODO test selecting half of parameters or half of the words
            dfs = dfs.reset_index()
            dfs = dfs.sort_values('tf_idf')
            
            last_resolutions = dfs[['resolution','tf_idf']].groupby('resolution').max()
            last_resolutions = last_resolutions.sort_values('tf_idf').index
            print('last set of resolutions:')
            print(last_resolutions)
            
            last_worst_resolutions = self._half_split(last_resolutions)
            if(last_worst_resolutions.size == dfs['resolution'].unique().size):
                raise 'The worst resolutions comprehense all the resolution \
                        isntead of half of resolutions'
                        
            self._transformer.remove_resolutions(last_worst_resolutions)

        return dfs
        
     
        
    def predict_proba(self):
        return .0
    
    def predict(self):
        return 0
    
    def _get_samples(self, sample_size, classes, labels):
        
        samples = pd.Series([])
        for c in classes:
            # TODO get classes column name
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



























