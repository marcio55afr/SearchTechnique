import random
import math
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
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
                 dimension_reduction_prop=1,
                 alphabet_size=4,
                 initial_sample_per_class=300,
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
        
        ## teste
        self._num_test_words = 0
        
        
        capabilities = {
            "multivariate": False,
            "unequal_length": False,
            "missing_values": False,
            "train_estimate": False,
            "contractable": False,
        }
        
    def fit(self, data, labels, n, window):
        
        # Check the data indexes
        if(not data.index.is_unique):
            index_ = range(data.shape[0])
            data.index = index_
            labels.index = index_
        
        # Check the number of samples and labels
        data_size,_ = data.shape
        num_labels = len(labels)
        if(data_size != num_labels):
            raise 'The number of samples and labels must be the same'
            
        # Getting the max length Series
        if(self.max_series_length is None):
            # All sample with same length
            self.max_series_length = data.iloc[0,0].size
        
        
                # TODO
        #-----------------------------------------------------------
        # First Paper - Experiments 
        # (window normalization X no window normalization)
        
        ######### normalization ############
        self._transformer = SaxNgram(self.max_series_length,
                                     #self.min_window_length,
                                     #self.window_prop,
                                     #self.dimension_reduction_prop,
                                     #self.alphabet_size,
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
        print('seed_list',seed_list)
        
        self._transformer.resolution._resolution_teste(n, window)
        
        ##################################################################
        iteration = -1
        while(True):
            iteration += 1
            samples_id, samples_label = self._get_samples_id(classes, labels, seed = seed_list[iteration])
            
            samples_size = samples_id.size
            print('\n\n\n\n\t\tIteration: {}'.format(iteration))
            print('sample size: {}'.format(samples_size))
            print('samples_id', samples_id)
            self._transformer.show_resolution()

            # Second Paper - technique ability
            # todo make the transformation multivariable!!!! important
            print('\nTransforming the series..')
            bags = self._transformer.transform(data.loc[samples_id,:], verbose=True)
            
            print('\nCalculating the tf of each word..')
            i=0
            dfs = pd.DataFrame([])
            for c in classes:
                samples_class_id = samples_label[samples_label==c].index
                # Second Paper - Algorithm speed up
                # todo separate the words from the meta informations for speed up
                #df = bags[samples_id]
                #df['sample_id'] = sample_id
                aux = bags.set_index('sample_id').loc[samples_class_id]
                df = aux.reset_index()
                resolutions = aux[['word', 'resolution']].groupby('word').head(1).set_index('word')
                df = aux.groupby('word').sum()

                df['resolution'] = resolutions
                df['label'] = c
                tf = df['frequency']/df.shape[0]
                df['tf'] = np.log( 1 + tf )
                if(not df.index.is_unique):
                    raise 'duplicated words detected inside the same class'
                
                # TODO
                # First Paper - Algorithm speed up
                # concat or append?
                dfs = dfs.append(df)
                
                i+=1

            print('Calculating the tf idf of each word..')
            # First Paper - Experiments
            # TODO remove from N the 'documents' from the same class
            # Try to consider all the samples within the same class as the a same
            # document
            dfs['tf_idf'] = 0
            doc_freq = pd.Series(Counter(dfs.index))
            dfs['doc_freq'] = doc_freq
            
            print('Calculating tf-idf by class')
            idf = np.log( classes.size/dfs['doc_freq'] )
            dfs['tf_idf'] = dfs['tf']*idf

            
            word_rank = dfs['frequency'].groupby('word').max()
            
            mean = word_rank.mean()
            #std = word_rank.std()
            
            #relevant_words = word_rank[word_rank > 7]
            

            ids = pd.Series(bags.sample_id.unique())
            bags = bags.set_index('word')
            #bags = bags.loc[ relevant_words.index ]
            
            x_train = pd.pivot_table(bags,
                                     index=['sample_id'],
                                     columns=['word'],
                                     values=['frequency'])
            x_train = x_train['frequency']
            sample_mask = ~ids.isin(x_train.index)
            x_train = x_train.append(pd.DataFrame(index=ids[sample_mask]))
            x_train = x_train.fillna(0)
            
            
            self._columns = x_train.columns
            self.simple_clf = KNeighborsClassifier(n_neighbors=5).fit(x_train, labels)
            '''self.simple_clf = LogisticRegression(solver='newton-cg',
                                                 multi_class='multinomial',
                                                 class_weight='balanced').fit(x_train, labels)'''
            print('Logistic Regression model trained.\n')
            self._is_fitted = True
            
            return self
            
            
            

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
            resolutions = dfs[['resolution','tf_idf']].groupby('resolution').mean()
            
            # TODO
            #-----------------------------------------------------------
            # First Paper - Experiments
            # remove (worst resolutions/ best resolutions / random resolutions)
            
            ######### worst resolutions ############
            '''resolutions = resolutions.sort_values('tf_idf').index'''
            ######### best resolutions ############
            resolutions = resolutions.sort_values('tf_idf', ascending=False).index
            ######### random resolutions ############
            '''resolutions = resolutions.index.to_series().sample(frac=1)'''
            
            #-----------------------------------------------------------
            
            resolutions_to_remove = self._data_split(resolutions, self._resolution_prop)
            if(resolutions_to_remove.size == dfs['resolution'].unique().size):
                raise 'The worst resolutions comprehense all the resolution \
                        isntead of a proportion of it'

            # First Paper - Technique Version !!! important
            # TODO Vote system to down or up vote features
            # necessarily to up vote in other versions of ST
            print('Resolutions to remove\n', resolutions_to_remove.to_list())
            resolutions_remains = self._transformer.remove_resolutions(resolutions_to_remove)

            # TODO
            #-----------------------------------------------------------
            # First Paper - Experiments
            # (break at the end / break at the middle)

            ######### break at the end ############
            '''if( samples_id.size == total_samples ):
                print('\n\nAll samples were processed!')
                break'''
            #-----------------------------------------------------------
            
            if((samples_size == data_size) or not resolutions_remains):
                break
                #return dfs
            
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
        x_data = self._extract_features(data)
        print('\n\nPredicting the data using the trained model\n')
        return self.simple_clf.predict(x_data)
    
    def _extract_predict_features(self, data):
        
        x_data = self._extract_features(data, True)
        x_data = x_data.append(pd.DataFrame([], columns=self._columns)).fillna(0)
        x_data.drop(x_data.index[-1])
        return x_data.loc[:,self._columns]
        
    def _extract_features(self, data, verbose=True):
        
        print('transforming the data...')
        bags = self._transformer.fit_transform(data)
        print('extracting features...\n')
        words = bags.word
        ids = pd.Series(bags.sample_id.unique())
        bags = bags.set_index('word')
        
        mask = self._columns.isin(words)
        self._num_test_words = mask.sum()
        bags = bags.loc[ self._columns[mask] ]
        
        print('\nTranposing the table to turn each word a feature')
        features = pd.pivot_table( bags[['sample_id','frequency']], index=['sample_id'], columns=['word'] )
        if(not features.empty):
            features = features['frequency']
        sample_mask = ~ids.isin(features.index)
        features = features.append(pd.DataFrame(index=ids[sample_mask]))
        features[ self._columns[~mask] ] = 0
        
        return features.fillna(0)
        
        v=0
        aux = bags.index.size//10
        print('\nTranposing the table to turn each word a feature')
        pd.pivot_table( bags[['sample_id','frequency']], index=['sample_id'], columns=['word'] )
        
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
        

    def _get_samples_id(self, classes, data_labels, seed=None):

        # Creating a list with the number of samples for each class to make
        # the sample
        sample_sizes=[]
        if(self.unbalanced_classes):
            sample_sizes = self._accumulated_multiplier*self._class_proportion
        else:
            s_size = self._accumulated_multiplier*self.initial_sample_per_class
            sample_sizes = [s_size]*classes.size
        
        # Sampling the data based on each class with reproducibility
        samples_label =  pd.Series([])
        for c,sample_size in zip(classes, sample_sizes):
            index = pd.Series(data_labels[data_labels==c].index)
            # if the sample is greater than the number of samples,
            # just take all the samples available
            if(sample_size > index.size):
                samples_label = samples_label.append(data_labels[index])
                continue
            
            s = index.sample(sample_size, random_state=seed)
            samples_label = samples_label.append(data_labels[s])
        
        return samples_label.index, samples_label
    
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
        random.seed(self.random_state)
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



























