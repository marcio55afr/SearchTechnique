import math
import numpy as np
import pandas as pd
from sktime.classification.base import BaseClassifier

from sax_ngram import SaxNgram
from ngram_features import NgramFeatures


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
        
        self.features = NgramFeatures(self.max_series_length,
                                 self.min_window_length,
                                 self.window_prop)
        
        # Variables
        self._transformer = SaxNgram(self.features,
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
            print('\n\n sample size: {}'.format(samples.size))
            
            bags = self._transformer.fit_transform(data.iloc[samples,:])
            dfs = pd.DataFrame([])
            
            i=0
            for bag in bags:
                
                df = pd.DataFrame.from_dict(bag, orient='index')
                df = df.reset_index()
                df['feature'] = [ ' '.join(word.split(' ')[0:2]) for word in df['index']]
                df.columns = ['word', 'frequency', 'feature']
                df['sample'] = i
                df['total'] = df.shape[0]
                
                dfs = pd.concat([dfs,df], axis=0, join='outer')
                
                i+=1
            
                        
            if( n_classes*sample_size > data_length ):
                _finished = True
            
            # TODO test this parameter of half parameters double samples
            sample_size *= 2
            #return dfs
        
    
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