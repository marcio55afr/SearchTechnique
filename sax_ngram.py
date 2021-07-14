# -*- coding: utf-8 -*-


from sktime.transformations.base import _PanelToPanelTransformer
from sktime.utils.validation.panel import check_X
from sktime.transformations.panel.dictionary_based import PAA

from ngram_resolution import NgramResolution

from math import floor
import numpy as np
import pandas as pd
import scipy.stats
import sys


__all__ = ["SaxNgram"]

class SaxNgram(_PanelToPanelTransformer):
    '''
    A version of the SAX (Symbolic Aggregate approXimation) Transformer that
    can create all n-gram words. This version works like the SAX described in
    Lin, Jessica, Eamonn Keogh, Li Wei, and Stefano Lonardi.
    "Experiencing SAX: a novel symbolic representation of time series."
    Data Mining and knowledge discovery 15, no. 2 (2007): 107-144.
    and change the parameters in order to get different words.
    Overview: its instance define the words and windows lengths and the size
    of the alphabet.
    The transform function runs for each series like that:
        define a list of window lengths to slide on the series;
        each window length create a list of words using the PAA approximation
        and the predefined breakpoints;
        each list create all n-gram features as possible;

    Parameters
    ----------
    minimum_window_length : int, optional
        Defines the shortest window length to slide across the series, this
        parameter must be lesser than the smallest series.
        The default is 6.
    maximum_window_prop : int, optional
        Defines the longest window length to slide acrros the series and it must
        be between [0,1].
        If 0 then only one window will slide acrros the series with length
        equal to the minimum_window_lentgh.
        The default is .50.
    dimension_reduction_prop : int, optional
        Defines the word proportion related to the window lentgh and it must
        be between (0,1]. The word lentgh is calculated by
        int(window_length*dimension_reduction_prop) and result must be greater
        than 1.
        The default is .80.
    alphabet_size : int, optional
        Defines the number of unique symbols to discretize the series.
        The default is 4.
    normalize : boolean, optional
        Defines if each window is normalized before the discretization.
        If False the breakpoints must be recalculate estimating the data
        distribution.
        The default is True.

    '''

    def __init__(self,
                 max_series_length,
                 min_window_length,
                 window_prop,
                 dimension_reduction_prop,
                 alphabet_size,
                 normalize=True):
        # Attributes
        self.max_series_length = max_series_length
        self.min_window_length = min_window_length
        self.window_prop = window_prop
        self.dimension_reduction_prop = dimension_reduction_prop
        self.alphabet_size = alphabet_size
        self.normalize = normalize
        self.resolution = NgramResolution(self.max_series_length,
                                          self.min_window_length,
                                          self.window_prop)
        
        # Local variables
        self._breakpoints = None
        self._alphabet = self._generate_alphabet()
        self._bin_symbols = floor(np.log2(self.alphabet_size)) + 1
        self._frequency_thereshold = 0
    
    def transform(self, data):
        '''
        Transforms a set of time series into a set of histograms(bag of words).

        Parameters
        ----------
        data : pandas.DataFrame
            Nested DataFrame containing series which will be discretized.

        Returns
        -------
        histograms : list
            List of dictionaries that counts the word frequencies in a time
            series.

        '''
        
        # Second Paper - technique ability
        # todo check for others types of data and handle each of them
        # todo decide process multivariate data or not!!! important
        data = check_X(data, enforce_univariate=True, coerce_to_pandas=True)
        data = data.squeeze(1)
        
        self._breakpoints = self._generate_breakpoints(data)
        
        # Variables
        histograms = pd.Series(index=data.index, dtype = object)
        
        # Counting the words for each sample of the data
        for sample_id in data.index:
            sample = data[sample_id]
            
            # Size of the sample (time series)
            series_length = sample.size
            
            # A Series of bag of words with all resolutions
            bag_of_bags = pd.DataFrame()
            
            # Multiple resolutions using various windows lenghts
            window_lengths = self.resolution.get_window_lengths_list(series_length)
            for window_length in window_lengths:                
                #print(window_length, end=' ')
                
                # Number of sliding windows inside the time series sample
                num_windows = series_length - window_length + 1

                # taking all sliding windows fixed on one set of parameter       
                windows = self._get_sliding_windows(sample, window_length, num_windows)

                # If the parameter normalize is true, each window will be normalized
                if(self.normalize):
                    windows = scipy.stats.zscore(windows, axis=1)
                
                # Creating a nested DataFrame to be transformed by the class PAA
                windows_df = pd.DataFrame()
                windows_df[0] = [pd.Series(x, dtype=np.float32) for x in windows]
                
                # Calculating the word length regarded by the window length
                word_length = int(window_length * self.dimension_reduction_prop)
                
                # Approximating each window and reducing its dimension
                paa = PAA(num_intervals=word_length)
                windows_appr = paa.fit_transform(windows_df)
                
                # Discretizing each window into a word
                words = windows_appr[0].apply(self._generate_bin_word)
                
                # Second Paper - Algorithm speed up
                # todo Optimizes to a array of string then use Counter to make a dictionary
                # Counting the frequency of each n-gram for each window length
                ngram_word_frequency = self._get_ngrams_word_count(words, window_length)
                bag_of_bags = bag_of_bags.append(ngram_word_frequency, ignore_index=True)
            
            # verify if all windows and all ngram only transform
            # a timeseries into features non-frequenty 
            if(not bag_of_bags.size):
                raise 'A sample was not able to be discretized. The remmaining resolutions produces only non-frequent words or the variable _frequency_thereshold is too high.'
            
            # Group the histograms of all samples
            histograms[sample_id] = bag_of_bags
        print('size of all bags: ', sys.getsizeof(bag_of_bags))
        return histograms
    
    def remove_resolutions(self, resolutions):
        self.resolution.remove(resolutions)
    
    def show_resolution(self):
        self.resolution.show()
        
    def _get_sliding_windows(self, sample, window_length, num_windows):
        
        return np.array(
                    sample[
                        np.arange(window_length)[None, :]
                        + np.arange(num_windows)[:, None],
                    ],
                    dtype= np.float32
                )

    def _get_ngrams_word_count(self, words, window_length):
        
        num_words = words.size
        bag_of_bags = pd.DataFrame()
        for n in self.resolution.get_ngrams_remaining(window_length):
            
            if((num_words -(n-1)*window_length) <= 0):
                break
            
            bag_of_ngrams = dict()
            for j in range(num_words -(n-1)*window_length ):
                # Second Paper - technique ability
                # todo assign on the feature its dimension id
                ngram = self._join_words(words.iloc[np.asarray(range(n))*window_length + j])
                #ngram = ' '.join(words[np.asarray(range(n))*window_length + j])
                bag_of_ngrams[ngram] = bag_of_ngrams.get(ngram,0) + 1
            
            resolution = '{} {}'.format(window_length, n)
            bag_of_ngrams = pd.DataFrame.from_dict(bag_of_ngrams, orient='index')
            bag_of_ngrams = bag_of_ngrams.reset_index()
            bag_of_ngrams['resolution'] = resolution
            bag_of_ngrams.columns = ['word','frequency','resolution']
            # TODO
            # First Paper - Experiments
            # (frequent words X resolution with frequent words X all words)
            
            ######### all words ############
            
            bag_of_bags = bag_of_bags.append(bag_of_ngrams, ignore_index=True)
            
            
            ######### resolution with frequent words ############
            # Verifies the existence of frenquenty features
            # if yes - adds all features into the bag of words of the
            '''
            if(any(bag_of_ngrams > self._frequency_thereshold)):
                bag_of_bags[resolution_id] = bag_of_ngrams
            '''
            ######### frequent words ############
            # Verifies the existence of frenquent features
            # and add only the frequent features to save memory
            # only in the proccess to find out the resolutions
            '''
            frequent_features = bag_of_ngrams > self._frequency_thereshold
            if(any(frequent_features)):
                bag_of_bags[resolution_id] = bag_of_ngrams
            '''
            
        if(bag_of_bags.size == 0):
            
            bag_of_ngrams = dict()
            for w in words:
                bag_of_ngrams[w] = bag_of_ngrams.get(w,0) + 1
            
            resolution = '{} 1'.format(window_length)
            bag_of_ngrams = pd.DataFrame.from_dict(bag_of_ngrams, orient='index')
            bag_of_ngrams = bag_of_ngrams.reset_index()
            bag_of_ngrams['resolution'] = resolution
            bag_of_ngrams.columns = ['word','frequency','resolution']
            
            bag_of_bags = bag_of_bags.append(bag_of_ngrams, ignore_index=True)
        return bag_of_bags
    
    def _join_words(self, words):
        
        if(not words.size):
            raise 'To join words the words must be sent to the function'
        
        ngram = words.iloc[0]
        for i in range(words.size - 1):
            ngram = ngram << self._bin_symbols # A sequence of 0' represents a space between ngrams
            ngram = (ngram << self._bin_symbols) | words.iloc[i+1]
        
        return ngram
    
    def _generate_word(self, window):
        '''
        Each value of the window is transformed into a alphabet letter, 
        this transformation depends on the breakpoints before stablished
        
        Parameters
        ----------
        window : list or array like
            The series window to be discretized, must be an interator
            with number values.

        Returns
        -------
        word : string
            Returns the corresponding word to the window considering
            the alphabet and the breakpoints.
        '''

        word = ''
        aux = ''
        # runs through the window discretizing one value at a time.
        for value in window:
            for bp in range(self.alphabet_size):
                if value <= self._breakpoints[bp]:
                    aux = self._alphabet[bp]
                    break
            word+=aux
        return word
        
    def _generate_bin_word(self, window):
        '''
        Each value of the window is transformed into a alphabet letter, 
        this transformation depends on the breakpoints before stablished
        
        Parameters
        ----------
        window : list or array like
            The series window to be discretized, must be an interator
            with number values.

        Returns
        -------
        word : int
            Returns the corresponding word to the window considering
            the alphabet and the breakpoints.
        '''

        word = 0
        # runs through the window discretizing one value at a time.
        for value in window:
            for bp in range(self.alphabet_size):
                if value <= self._breakpoints[bp]:
                    #aux = self._alphabet[bp]
                    word = (word << self._bin_symbols) | (bp+1)
                    break
                
        return word

    def _generate_breakpoints(self,data=None):
        
        if(self.normalize):
            # Pre-made gaussian curve breakpoints from UEA TSC codebase
            return {
                2: [0, sys.float_info.max],
                3: [-0.43, 0.43, sys.float_info.max],
                4: [-0.67, 0, 0.67, sys.float_info.max],
                5: [-0.84, -0.25, 0.25, 0.84, sys.float_info.max],
                6: [-0.97, -0.43, 0, 0.43, 0.97, sys.float_info.max],
                7: [-1.07, -0.57, -0.18, 0.18, 0.57, 1.07, sys.float_info.max],
                8: [-1.15, -0.67, -0.32, 0, 0.32, 0.67, 1.15, sys.float_info.max],
                9: [-1.22, -0.76, -0.43, -0.14, 0.14, 0.43, 0.76, 1.22, sys.float_info.max],
                10: [-1.28, -0.84, -0.52, -0.25, 0.0, 0.25, 0.52, 0.84, 1.28, sys.float_info.max],
            }[self.alphabet_size]
        
        else:
            s = pd.Series()
            for time_series in data:
                s = s.append(time_series)
                
            s = s.sort_values()
            bucket_size = s.size//self.alphabet_size
            breakpoints = []
            
            for bucket in range(self.alphabet_size-1):
                bp_id = bucket_size*(bucket+1)
                bp = s[bp_id:bp_id+2].sum()/2
                breakpoints.append(bp)
            breakpoints.append(sys.float_info.max)

    def _generate_alphabet(self):
        # Symbols of an anphabet to be used in the discretization
        return {
            2: ['a', 'b'],
            3: ['a', 'b','c'],
            4: ['a', 'b','c','d'],
            5: ['a', 'b','c','d','e'],
            6: ['a', 'b','c','d','e','f'],
            7: ['a', 'b','c','d','e','f','g'],
            8: ['a', 'b','c','d','e','f','g','h'],
            9: ['a', 'b','c','d','e','f','g','h','i'],
            10: ['a', 'b','c','d','e','f','g','h','i','j'],
        }[self.alphabet_size]

    def _generate_window_lengths(self, series_length):
        
        # If the minimum window is longer than the series raise an error
        if(self.minimum_window_length > series_length):
            raise 'The series has length of {} and \
            it is shorter than minimum window of {} length'.format(self.minimum_window_length , series_length)
                
        aux = self.minimum_window_length
        max_length = self.maximum_window_prop * series_length
        window_lengths = [aux]
        aux *= 2
        
        while(aux <=  max_length):
            window_lengths.append(aux)
            aux *= 2

        return window_lengths
    


















