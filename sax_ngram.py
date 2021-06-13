# -*- coding: utf-8 -*-


from sktime.transformations.base import _PanelToPanelTransformer
from sktime.utils.validation.panel import check_X
from sktime.transformations.panel.dictionary_based import PAA


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
                 features,
                 min_window_length,
                 max_window_prop,
                 dimension_reduction_prop,
                 alphabet_size,
                 normalize=True):
        # Attributes
        self.min_window_length = min_window_length
        self.max_window_prop = max_window_prop
        self.dimension_reduction_prop = dimension_reduction_prop
        self.alphabet_size = alphabet_size
        self.normalize = normalize
        
        # Local variables
        self.features = features
        self._breakpoints = []
        self._alphabet = []
    
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
        
        # TODO check for others types of data and handle each of them
        data = check_X(data, enforce_univariate=True, coerce_to_pandas=True)
        data = data.squeeze(1)
        
        # Variables
        self._breakpoints = self._generate_breakpoints()
        self._alphabet = self._generate_alphabet()
        histograms = []
                
        # Counting the words for each sample of the data
        for sample in data:
            
            # Variable size to each sample of the data
            series_length = sample.size
            
            # Bag of words of each time series
            histogram = dict()
            
            # Multiple resolutions using various windows lenghts
            window_lengths = self.features.get_window_lengths_list(series_length)
            for window_length in window_lengths:
                
                print(window_length, end=' ')
            
                # taking all sliding windows fixed on one set of parameter       
                num_windows = series_length - window_length + 1
                windows = np.array(
                    sample[
                        np.arange(window_length)[None, :]
                        + np.arange(num_windows)[:, None],
                    ],
                    dtype= np.float32
                )
                
                # TODO if normalize == False then breakpoints must change
                #   iterativily calcule the mean and the standard deviation
                #   and calcule the breakpoints in order to get equiprobability

                # If the parameter normalize is true, each window will be normalized
                if(self.normalize):
                    windows = scipy.stats.zscore(windows, axis=1)
                
                # Creating a nested DataFrame to be transformed by the class PAA
                windows_df = pd.DataFrame()
                windows_df[0] = [pd.Series(x, dtype=np.float32) for x in windows]
                
                # word length for each window length
                word_length = int(window_length * self.dimension_reduction_prop)
                
                # Approximating each window and reducing its dimension
                paa = PAA(num_intervals=word_length)
                windows_appr = paa.fit_transform(windows_df)
                windows_appr = np.asarray([np.asarray(a) for a in windows_appr.iloc[:, 0]])
                
                # Discretizing each window into a word
                words = [self._create_word(window) for window in windows_appr]
                
                # TODO n-grams without superposition
                # TODO Optimizes to a array of string the use Counter to make a dictionary
                # Counting the frequency of each n-gram for each window length  
                for n in self.features.get_ngrams_remaining(window_length):
                    dict_aux = dict()
                    for i in range(num_windows -n +1):
                        feature_id = [str(word_length), str(n)]
                        ngram = ' '.join(feature_id+words[i:i+n])
                        dict_aux[ngram] = dict_aux.get(ngram,0) + 1
                    _has_frequent_features = (np.asarray(list(dict_aux.values()))>2).any()
                    if(_has_frequent_features):
                        for key, value in dict_aux.items():
                            histogram[key] = value
                
            
            print('size of the bag: ', sys.getsizeof(histogram))
            # Group the histograms of all samples
            histograms.append(histogram)
        return histograms
    
    def _create_word(self, window):
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
                    #word = (word << 2) | bp
                    break
            word += aux

        return word

    def _generate_breakpoints(self):
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
            10: [
                -1.28,
                -0.84,
                -0.52,
                -0.25,
                0.0,
                0.25,
                0.52,
                0.84,
                1.28,
                sys.float_info.max,
            ],
        }[self.alphabet_size]

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


















