import random
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
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
                 max_series_length = None,
                 min_window_length = 6,
                 window_prop = .5,
                 dimension_reduction_prop=.8,
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
        self.random_state = random_state
        self.simple_clf = None
        
        # TODO
        #-----------------------------------------------------------
        # First Paper - Experiments (unbalanced X balanced samples)
        
        ######### balanced samples ############
        self.unbalanced_classes = False
        
        ######### unbalanced samples ############
        '''self.unbalanced_classes = True'''
        #------------------------------------------------------------
        
        

        
        # TODO
        #-----------------------------------------------------------
        # First Paper - Experiments
        # test the proportions
        # ( x1.5 x0.75 / x2 x0.5 / x3 x0.333 )
        
        ######### proportion 1 ############
        '''self._sample_multiplier = 1.5
        self._resolution_prop = 2/3'''
        
        ######### proportion 2 ############
        self._sample_multiplier = 2
        self._resolution_prop = 1/2
        
        ######### proportion 3 ############
        '''self._sample_multiplier = 3
        self._resolution_prop = 1/3'''
        #-----------------------------------------------------------
        
        
        # Variables
        self._transformer = None
        self._accumulated_multiplier = 1
        self._class_proportion = pd.Series()
        self._max_class_group = None
        self._iteration_limit = None
        self._random_list = []
        self._columns = None
        
    def fit(self, data, labels):
        
        # Check the data indexes
        if(not data.index.is_unique):
            index_ = range(data.shape[0])
            data.index = index_
            labels.index = index_
        
        # Check the number of samples and labels
        total_samples,_ = data.shape
        total_labels = len(labels)
        if(total_samples != total_labels):
            raise 'The number of samples and labels must be the same'
            
        # Getting the max length Series
        if(self.max_series_length is None):
            self.max_series_length = self._get_max_length(data)
        
        
                # TODO
        #-----------------------------------------------------------
        # First Paper - Experiments 
        # (window normalization X no window normalization)
        
        ######### normalization ############
        self._transformer = SaxNgram(self.max_series_length,
                                     self.min_window_length,
                                     self.window_prop,
                                     self.dimension_reduction_prop,
                                     self.alphabet_size,
                                     normalize = True)
        
        ######### non normalization ############
        '''
        self._transformer = SaxNgram(self.max_series_length,
                                     self.min_window_length,
                                     self.window_prop,
                                     self.dimension_reduction_prop,
                                     self.alphabet_size,
                                     normalize = False)
        '''
        #-----------------------------------------------------------
            
        
        ################ Initializating the iteration ####################

        classes = labels.unique()
        
        # Calculating the class proportion
        self._generate_class_proportion(labels, classes)
        
        # Calculating the max number of iterations
        self._generate_iteration_limit(labels, classes)
        
        # Generating a list of random to sample the data
        seed_list = self._get_random_list()
        
        ##################################################################
        iteration = -1
        while(True):
            iteration += 1
            samples_id = self._get_samples_id(classes, labels, seed = seed_list[iteration])
            
            N = samples_id.size
            print('\n\n\n\n\t\tIteration: {}'.format(iteration))
            print('sample size: {}'.format(N))
            self._transformer.show_resolution()

            # Second Paper - technique ability
            # todo make the transformation multivariable!!!! important
            print('\nTransforming the series..')
            bags = self._transformer.transform(data.loc[samples_id,:], verbose=True)
            
            print('\nCalculating the tf of each word..')
            i=0
            dfs = pd.DataFrame([])
            for sample_id in samples_id:
                # Second Paper - Algorithm speed up
                # todo separate the words from the meta informations for speed up
                df = bags[sample_id]
                df['sample_id'] = sample_id
                total = df.shape[0]
                df['tf'] = df['frequency']/total
                
                # TODO
                # First Paper - Algorithm speed up
                # concat or append?
                dfs = pd.concat([dfs,df], axis=0, join='outer')
                
                i+=1

            print('Calculating the tf idf of each word..')
            # First Paper - Experiments
            # TODO remove from N the 'documents' from the same class
            # Try to consider all the samples within the same class as the a same
            # document
            dfs['tf_idf'] = 0
            doc_freq = pd.Series(Counter(dfs['word']))
            dfs = dfs.set_index('word')
            dfs['doc_freq'] = doc_freq
            i=0
            i_max = samples_id.size
            for sample in samples_id:
                i+=1
                if(i > (i_max/10)):
                    print('#',end=' ')
                    i -= i_max//10

                df = dfs[dfs.sample_id == sample]
                tf = df['tf']
                idf = np.log2(N/(df['doc_freq']+1))                
                dfs.loc[dfs.sample_id == sample,'tf_idf'] = tf*idf

            #-----------------------------------------------------------
            ######### break at the middle ############
            
            #-----------------------------------------------------------
            
            self._accumulated_multiplier *= self._sample_multiplier

            dfs = dfs.reset_index()
            dfs = dfs.sort_values('tf_idf')

            print('\n\nRemoving worst resolutions found in this iteration')
            # First Paper - Experiments
            # TODO compare half of parameters against half of the words
            # caution!
            # half of the best words could comprehend all the resolutions!
            resolutions = dfs[['resolution','tf_idf']].groupby('resolution').max()
            resolutions = resolutions.sort_values('tf_idf').index

            worst_resolutions = self._data_split(resolutions, self._resolution_prop)
            if(worst_resolutions.size == dfs['resolution'].unique().size):
                raise 'The worst resolutions comprehense all the resolution \
                        isntead of a proportion of it'

            # First Paper - Technique Version !!! important
            # TODO Vote system to down or up vote features
            # necessarily to up vote in other versions of ST
            print('Wors Resolutions\n', worst_resolutions.to_list())
            self._transformer.remove_resolutions(worst_resolutions)
            self._transformer.show_resolution()

            # TODO
            #-----------------------------------------------------------
            # First Paper - Experiments
            # (break at the end / break at the middle)

            ######### break at the end ############
            '''if( samples_id.size == total_samples ):
                print('\n\nAll samples were processed!')
                break'''
            #-----------------------------------------------------------
            
            if(samples_id.size == total_samples):
                #return dfs
                break
            
        print('\n\nFeature search completed!!\n\n')
        
        print('Starting the traing process of the Logistic Regression classifier\n')
        x_train = self._extract_features(data)
        self._columns = x_train.columns
        self.simple_clf = LogisticRegression(solver='newton-cg',
                                             multi_class='multinomial',
                                             class_weight='balanced').fit(x_train, labels)
        print('Logistic Regression model trained.\n')
        self._is_fitted = True
        
        return self
        
    def predict_proba(self, data):
        
        # Check the data indexes
        if(not data.index.is_unique):
            index_ = range(data.shape[0])
            data.index = index_
        
        print('\nExtracting the features from this new data')
        x_data = self._extract_predict_features(data)
        print('\nPredicting the data probability using the trained model\n')
        return self.simple_clf.predict_proba(x_data)
    
    def predict(self, data):
        
        # Check the data indexes
        if(not data.index.is_unique):
            index_ = range(data.shape[0])
            data.index = index_
        
        print('\nExtracting the features from this new data')
        x_data = self._extract_predict_features(data)
        print('\n\nPredicting the data using the trained model\n')
        return self.simple_clf.predict(x_data)
    
    def _extract_predict_features(self, data):
        
        x_data = self._extract_features(data, True)
        x_data = x_data.append(pd.DataFrame([], columns=self._columns)).fillna(0)
        x_data.drop(x_data.index[-1])
        return x_data.loc[:,self._columns]
        
    def _extract_features(self, data, verbose=True):
        
        print('extracting features...\n')
        features = pd.DataFrame(columns=self._columns)
        print('transforming the data...')
        bags = self._transformer.fit_transform(data)

        v=0
        aux = bags.index.size//10
        print('\nTranposing the table to turn each word a feature')
        for bag_id in bags.index:
            bag = bags[bag_id]
            df = bag[['word','frequency']].T
            df.columns = df.loc['word']
            df = df.drop('word')
            df.index = [bag_id]
            df.index.name = 'index'
            features = features.append(df)
            
            v+=1
            if(verbose):
                if(v==aux):
                    print('#',end='')
                    v=0
        
        # Perhaps returns the transposition of the table would be better
        return features.fillna(0)
    
    def _extract_features_from_dfs(self, dfs):
        
        features = pd.DataFrame()
        for sample_id in dfs['sample_id']:
            df = dfs.loc[0,['word','frequency']].T
            df.columns = df.loc['word']
            df = df.drop('word')
            df.index = [sample_id]
            features = features.append(df)
        
        return features

    def _get_max_length(self, data):
        
        max_length = 0
        for col in data.columns:
            for timeseries in data[col]:
                if(timeseries.size > max_length):
                    max_length = timeseries.size
        
        return max_length
        

    def _get_samples_id(self, classes, labels, seed=None):

        # Creating a list with the number of samples for each class to make
        # the sample
        sample_sizes=[]
        if(self.unbalanced_classes):
            sample_sizes = self._accumulated_multiplier*self._class_proportion
        else:
            s_size = self._accumulated_multiplier*self.initial_sample_per_class
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
        
        if(self.initial_sample_per_class < 1):
            raise 'The smallest sample is 2 samples per classes of the data.'
        
        if(self._max_class_group is None):
            raise 'to calculate the number of iteration the class proportion must be calculated first'
     
        self._iteration_limit = math.ceil(np.log2(self._max_class_group)) + 2
    
    def _get_random_list(self):
        
        if(self._iteration_limit is None):
            self._generate_iteration_limit()
            
        num_elements = self._iteration_limit
        rand_list = [random.randint(100,1000) for _ in range(num_elements)]        
        return rand_list
        
    def _get_histogram(self):
        return pd.DataFrame([])
    
    def _data_split(self, data, proportion, part='first'):
        
        split_point = int(data.size*proportion)
        
        if(part=='first'):
            return data[:split_point]
        
        elif(part=='second'):
            return data[split_point:]

        raise 'The function _half_split can only return the first or second part'



























