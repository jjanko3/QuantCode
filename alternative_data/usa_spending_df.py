# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 16:11:21 2019

@author: jjanko
"""

import pandas as pd
import os


rootDir = r'D:\usa_spending'

for dirName, subdirList, fileList in os.walk(rootDir):
    pass



for i in fileList:
    if 'csv' in i:
        data = pd.read_csv(os.path.join(rootDir, i))
        data.to_hdf(os.path.join(rootDir, i.replace('.csv','') + '.h5'), key = 'df')
    
        
        
columns_list = pd.read_excel(r'D:\columns_contracts.xlsx')

columns_list = list(columns_list[columns_list.columns[0]])

df = pd.DataFrame()

for i in fileList:
    if '.csv' in i:
        data = pd.read_csv(os.path.join(rootDir, i))
        data = data[columns_list]
        data.to_hdf(os.path.join(r'D:\usa_spending_clean', i.replace('.csv','') + '.h5'), key = 'df')
    



