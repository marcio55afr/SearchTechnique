# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sktime.transformations.panel.dictionary_based import SAX, SFA, PAA
from sktime.utils.validation.panel import check_X
import scipy.stats


from search_technique import SearchTechnique


df_train = pd.read_hdf('AtrialFibrillation', key='train')

ts1 = df_train.iloc[0,0].values
ts2 = df_train.iloc[1,0].values

time_series =  pd.DataFrame([[ts1]])
print(time_series)
st = SearchTechnique(640)
fit = st.fit(time_series, [0])
print(sum(fit[0].values()))

n_features = len(fit[0].values())

bow = pd.DataFrame(sorted(fit[0].values(), reverse=True))
bow = bow.reset_index()
bow.columns = ['Word index', 'Word Frequency']

bow.plot.scatter(x='Word index', y='Word Frequency')