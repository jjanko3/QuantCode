# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 10:53:11 2019

@author: jjanko
"""
import pandas as pd
import os
import dask.dataframe as dd


rootDir = r'D:\usa_spending_clean'
for dirName, subdirList, fileList in os.walk(rootDir):
    pass


df = pd.DataFrame()

for i in fileList:
    data = pd.read_hdf(os.path.join(rootDir, i),index_col=False )
    if len(df) == 0:
        df = data.copy()
    else:
        df = pd.concat([df, data], axis = 0)
    