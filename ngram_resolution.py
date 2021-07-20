# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from math import ceil

class NgramResolution(object):
    
    def __init__(self, min_window_size, max_window_size):
        
        if( max_window_size < min_window_size ):
            raise 'data with series length too short'
            
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        #self.window_prop = window_prop
        self.max_ngrams = max_window_size//min_window_size

            
        windows_diff = ceil((max_window_size - min_window_size)/10)
        window_sizes = np.arange(min_window_size, max_window_size+1, windows_diff)
                
        self.resolutions_matrix = pd.DataFrame(0,
                                              index=np.arange(1,self.max_ngrams+1),
                                              columns = window_sizes,
                                              dtype=bool)
        
        for window in window_sizes:
            ngram = (self.max_window_size//window)
            self.resolutions_matrix.loc[0:ngram,window] = True
            
    def get_ngrams_remaining(self, window_length):
        
        remaining = self.resolutions_matrix[window_length] == 1
        return self.resolutions_matrix.loc[remaining, window_length].index
    
    def get_window_lengths_list(self, series_length):
        
        aux = self.resolutions_matrix.sum()
        windows = aux[aux>=1].index.to_list()
        return windows
        mask = windows <= series_length*self.window_prop
        
        selected_windows = windows[mask].to_list()
        if(len(selected_windows)):
            return selected_windows
        if(series_length < windows[0]):
            raise 'The time series is shorter than the smallest window, remove this sample or decrease the smallest window'
        return [windows[0]]
    
    def remove(self, resolutions):
        
        if(self.resolutions_matrix.values.sum() <= 2):
            return
        
        for resolution in resolutions:
            window_length, ngram_length = resolution.split(' ')
            window_length = int( window_length)
            ngram_length = int(ngram_length)
            self.resolutions_matrix.loc[ngram_length, window_length] = 0
            
    def show(self):
        
        print('\nResolution Matrix')
        if(self.resolutions_matrix.shape[0] > 10):
            i = [0,1,2,3,4,5,6,7,8,-9,-8,-7,-6,-5,-4,-3,-2,-1]
            print(self.resolutions_matrix.iloc[i])
        else:
            print(self.resolutions_matrix)
            
        
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
               
    def _resolution_teste(self):
        
        for col in self.resolutions_matrix.columns:
            self.resolutions_matrix[col] = 0
        
        self.resolutions_matrix.iloc[0,0] = 1
               