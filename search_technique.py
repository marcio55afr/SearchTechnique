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
        self._proportion_list = pd.Series()
        self._max_class_group = None
        self._iteration_limit = None
        self._random_list = []
        
    def fit(self, data, labels):
        
        # Check the number of samples and labels
        total_samples,_ = data.shape
        total_labels = len(labels)
        if(total_samples != total_labels):
            raise 'The number of samples and labels must be the same'
            
        
        ################ Initializating the iteration ####################

        classes = labels.unique()
        
        # Calculating the class proportion
        self._generate_class_proportion(labels, classes)
        
        # Calculating the max number of iterations
        self._generate_iteration_limit(labels, classes)
        
        # Generating a list of random to sample the data
        seed_list = self._generate_random_list(labels, classes)
        
        #
        
        ##################################################################
        iteration = -1
        while(True):
            iteration += 1

            # TODO
            # First Paper - Experiments (Unbalanced vs Balanced samples)
            samples_id = self._get_samples_id(classes, labels, seed = seed_list[iteration])
            N = samples_id.size
            print('\nTransforming the series..')
            self._transformer.show_resoltuion()
            print('\nsample size: {}'.format(N))
            
            # Second Paper - technique ability
            # todo make the transformation multivariable!!!! important
            bags = self._transformer.fit_transform(data.iloc[samples_id,:])
            dfs = pd.DataFrame([])
            
            print('\nCalculating the tf of each word..\n')
            i=0
            for sample in samples_id:
                # Second Paper - Algorithm speed up
                # todo separate the words from the meta informations for speed up
                df = bags[sample]
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
            # First Paper - Experiments
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
            
            # First Paper - Experiments
            # TODO test this parameter of half parameters double samples
            # with other proportions like 1/3 3* or sum with fixed buckets
            self._sample_multiplier *= self._sample_multiplier

            dfs = dfs.reset_index()
            dfs = dfs.sort_values('tf_idf')

            print('\nRemoving worst resolutions found in this iteration')
            # First Paper - Experiments
            # TODO compare half of parameters against half of the words
            last_resolutions = dfs[['resolution','tf_idf']].groupby('resolution').max()
            last_resolutions = last_resolutions.sort_values('tf_idf').index
            #print('last set of resolutions:')
            #print(last_resolutions)
            
            last_worst_resolutions = self._half_split(last_resolutions)
            if(last_worst_resolutions.size == dfs['resolution'].unique().size):
                raise 'The worst resolutions comprehense all the resolution \
                        isntead of half of resolutions'
            
            # First Paper - Technique Version !!! important
            # TODO Vote system to down or up vote features
            # necessarily to up vote in other versions of ST
            self._transformer.remove_resolutions(last_worst_resolutions)

        return dfs
        
     
        
    def predict_proba(self):
        return .0
    
    def predict(self):
        return 0
    
    def _get_samples_id(self, classes, labels, seed=None):

        # Creating a list with the number of samples for each class to make
        # the sample
        sample_sizes=[]
        if(self.unbalanced_classes):
            sample_sizes = self._sample_multiplier*self._class_proportion
        else:
            s_size = self._sample_multiplier*self.initial_sample_per_class
            sample_sizes = [s_size]*classes.size
        
        # Sampling the data based on each class with reproducibility
        samples_id = pd.Series([])
        for c,sample_size in zip(classes, sample_sizes):
            index = pd.Series(labels[labels==c].index)
            # if the sample is greater than the number of samples,
            # just take all the samples available
            if(sample_size > index.size):
                samples_id = samples_id.append(index)
                continue
            
            s = index.sample(sample_size, random_state=seed)
            samples_id = samples_id.append(s)
        
        return samples_id
    
    def _generate_class_proportion(self, labels, classes, rewrite = False):
        
        if(self._class_proportion.size and not rewrite):
            print('_class_proportion was already calculated, set the option \
                  rewrite=True for rewriting')
        
        class_groups = pd.Series()
        
        for c in classes:
            class_groups[c] = labels[labels==c].size

        self._max_class_group = class_groups.max()
        total_samples = labels.size
        
        class_prop = class_groups/total_samples
        self._class_proportion = (class_prop*self.initial_sample_per_class)//class_prop.min()
        
    def _generate_iteration_limit(self, labels, classes):
        
        if(self._max_class_group is None):
            raise 'to calculate the number of iteration the class proportion \
                    must be calculated first'
     
        self._iteration_limit = np.log2(self._max_class_group)
    
    def _generate_random_list(self, num_elements):
        
        random.seed(self.random_state)
        return [random.random() for _ in range(num_elements)]
        
    def _get_histogram(self):
        return pd.DataFrame([])
    
    def _half_split(self, data, part='first'):
        
        half = len(data)//2
        
        if(part=='first'):
            return data[:half]
        
        elif(part=='second'):
            return data[half:]

        raise 'The function _half_split can only return the first or second part'



























