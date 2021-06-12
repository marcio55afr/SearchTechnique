# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


class NgramFeatures(object):
    
    def __init__(self, max_series_length, min_window_length, window_prop):
        
        self.min_window_length = min_window_length
        self.window_prop = window_prop
        self.max_window_length = max_series_length*window_prop
        self.max_ngrams = self.max_window_length//min_window_length
        
        # Convert the number to int
        self.max_window_length = int(self.max_window_length)
        self.max_ngrams = int(self.max_ngrams)

        window_lengths_list = self._generate_window_lengths(self.min_window_length,
                                                            self.max_window_length)
                
        self.parameters_matrix = pd.DataFrame(-1,
                                              index=np.arange(1,self.max_ngrams),
                                              columns = window_lengths_list,
                                              dtype=int )
        
        for window in window_lengths_list:
            num_ngrams_max = (self.max_window_length//window)
            self.parameters_matrix.loc[0:num_ngrams_max,window] = 1
            
    def get_ngrams_remaining(self, window_length):
        
        remaining = self.parameters_matrix[window_length] == 1
        return self.parameters_matrix.loc[remaining,
                                          window_length].index
    
    def get_window_lengths_list(self, series_length):
        
        windows = self.parameters_matrix.columns
        selecting_windows = windows <= series_length*self.window_prop
        return windows[selecting_windows].to_list()
        
    def _generate_window_lengths(self,min_length ,max_length):
        
        # If the minimum window is longer than the series raise an error
        if(min_length > max_length):
            raise 'The series has length of {} and \
            it is shorter than minimum window of {} length'.format(min_length,
            max_length)

        aux = min_length
        window_lengths = [aux]
        aux *= 2
        
        while(aux <=  max_length):
            window_lengths.append(aux)
            aux *= 2

        return window_lengths
               
               