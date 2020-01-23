# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:08:06 2020

@author: jjanko
"""


import pandas as pd
import pickle
import glob
import os

rootDir = r'D:\usa_spending_clean'


contract_file = 'all_contracts.csv'


df = pd.DataFrame()

in_path = os.path.join(rootDir, contract_file) #Path where the large file is
out_path = r'D:\cached' #Path to save the pickle files to
chunk_size = 100000 #size of chunks relies on your available memory
separator = ","

reader = pd.read_csv(in_path,sep=separator,chunksize=chunk_size, 
                    low_memory=False)    


for i, chunk in enumerate(reader):
    out_file = out_path + "/data_{}.pkl".format(i+1)
    with open(out_file, "wb") as f:
        pickle.dump(chunk,f,pickle.HIGHEST_PROTOCOL)
        

pickle_path = out_path #Same Path as out_path i.e. where the pickle files are

for dirName, subdirList, fileList in os.walk(pickle_path):
    pass

df = pd.DataFrame([])
for i in range(len(fileList)):
    df = df.append(pd.read_pickle(os.path.join(out_path,fileList[i])),ignore_index=True)