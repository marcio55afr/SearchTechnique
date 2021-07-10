import random
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
    
    unbalanced_classes:   boolean, default = True   
                    define the proportion of samples with different classes to
                    create a group of samples. If False, this group will have
                    initial_sample_size * number of classes inside the data.
                    If True, the proportion of classes in group will be the same
                    as in data. The smallest number of a class is defined by
                    initial_sample_size.
    
    random_state:   int, default = None
                    seed for the random functions.
                    
    '''


    def __init__(self,
                 max_series_length,
                 min_window_length = 6,
                 window_prop = .5,
                 dimension_reduction_prop=.80,
                 alphabet_size=4,
                 initial_sample_per_class=2,
                 unbalanced_classes = True,
                 random_state=None):
        
        self.max_series_length = max_series_length
        self.min_window_length = min_window_length
        self.window_prop = window_prop
        self.dimension_reduction_prop = dimension_reduction_prop
        self.alphabet_size = alphabet_size
        self.initial_sample_per_class = initial_sample_per_class
        self.unbalanced_classes = unbalanced_classes
        self.random_state = random_state
        
        # Variables
        self._transformer = SaxNgram(self.max_series_length,
                                     self.min_window_length,
                                     self.window_prop,
                                     self.dimension_reduction_prop,
                                     self.alphabet_size)
        self._sample_multiplier = 2
        self._iteration_limit = None
        
    def fit(self, data, labels):
        
        # Check the number of samples and labels
        total_samples,_ = data.shape
        total_labels = len(labels)
        if(total_samples != total_labels):
            raise 'The number of samples and labels must be the same'
            
        
        # Initializating the iteration
        classes = labels.unique()
        sample_size = self.initial_sample_per_class
        
        # Calculating the number of iterations
        self._calcule_iteration_limit(labels, classes)
        
        seed_list = self._generate_random_list(labels, classes)

        iteration = 0
        while(True):
            iteration += 1

            # TODO unbalanced classes
            samples_id = self._get_samples_id(sample_size, classes, labels, seed = seed_list[iteration-1])
            N = samples_id.size
            print('\nTransforming the series..')
            self._transformer.show_resoltuion()
            print('\nsample size: {}'.format(N))
            
            # TODO make the transformation multivariable!!!! important
            bags = self._transformer.fit_transform(data.iloc[samples_id,:])
            dfs = pd.DataFrame([])
            
            print('\nCalculating the tf of each word..\n')
            i=0
            for sample in samples_id:
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
            
            print('\nBag of Words')
            print(dfs[['word','frequency']])
            print('\nCalculating the tf idf of each word..')
            # TODO remove from N the 'documents' from the same class
            dfs['tf_idf'] = 0
            docs_freqs = pd.Series(Counter(dfs['word']))
            i=0
            i_max = samples_id.size
            for sample in samples_id:
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

            if( samples_id.size == total_samples ):
                print('\n\nAll samples were processed!')
                break
            
            # TODO test this parameter of half parameters double samples
            # with other proportions like 1/3 3* or sum with fixed buckets
            sample_size *= self._sample_multiplier

            dfs = dfs.reset_index()
            dfs = dfs.sort_values('tf_idf')

            print('\nRemoving worst resolutions found in this iteration')
            # TODO compare half of parameters against half of the words
            last_resolutions = dfs[['resolution','tf_idf']].groupby('resolution').max()
            last_resolutions = last_resolutions.sort_values('tf_idf').index
            #print('last set of resolutions:')
            #print(last_resolutions)
            
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
    
    def _get_samples_id(self, sample_size, classes, labels, seed=None):
        
        samples_id = pd.Series([])
        for c in classes:
            index = pd.Series(labels[labels==c].index)
            if(sample_size > index.size):
                samples_id = samples_id.append(index)
                continue
            
            s = index.sample(sample_size, random_state=seed)
            samples_id = samples_id.append(s)
        
        return samples_id
    
    def _generate_random_list(self, num_elements):
        
        random.seed(self.random_state)
        return [random.random() for _ in range(num_elements)]
        
        
    def _generate_iteration_limit(self, labels, classes):
        
        max_class_group = 0
        for c in classes:
            size_c = labels[labels==c].size
            if(max_class_group < size_c):
                max_class_group =  size_c
        
        self._iteration_limit = np.log2(max_class_group)
        
    def _get_histogram(self):
        return pd.DataFrame([])
    
    def _half_split(self, data, part='first'):
        
        half = len(data)//2
        
        if(part=='first'):
            return data[:half]
        
        elif(part=='second'):
            return data[half:]

        raise 'The function _half_split can only return the first or second part'



























