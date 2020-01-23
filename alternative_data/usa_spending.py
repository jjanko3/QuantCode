# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 13:45:38 2019

@author: jjanko
"""

import requests, zipfile, io
import os


for i in range(2001,2009):
    

    url = r'https://files.usaspending.gov/award_data_archive/' + str(i) + r'_all_Contracts_Full_20191210.zip'
    
    target_path = str(i) + '.zip'
    
    response = requests.get(url, stream=True)
    handle = open(target_path, "wb")
    for chunk in response.iter_content(chunk_size=512):
        if chunk:  # filter out keep-alive new chunks
            handle.write(chunk)
    handle.close()
    
    
    
for i in range(2001,2020):
    import zipfile
    with zipfile.ZipFile(os.path.join(os.getcwd(),str(i) + '.zip'), 'r') as zip_ref:
        zip_ref.extractall(r'D:\usa_spending')
        