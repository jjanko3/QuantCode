# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 15:15:13 2019

@author: jjanko
"""

import os
import pandas as pd
import numpy as np
from statsmodels import regression



if __name__ == "__main__":
    
    
    #data = pd.read_hdf(os.path.join(r'T:\PhD Students','hedonic_data.h5'))
    data=pd.read_hdf(os.path.join(r'T:\PhD Students\hedonic_data','hedonic_data.h5'))
    

    #create data for unit type fixed effect
    data['Unit Type'] = 'None' 
    data.loc[data['CONDO'] == 1, 'Unit Type'] = 'Condo'

    data.loc[(data['year'] < 2015) & (data['NUNIT2'] == 1 ),'Unit Type'] = 'Detached'
    data.loc[(data['year'] < 2015) & (data['NUNIT2'] == 2 ),'Unit Type'] = 'Attached'
    
    
    data.loc[(data['year'] >= 2015) & (data['TYPE'] == 2 ),'Unit Type'] = 'Detached'
    data.loc[(data['year'] >= 2015) & (data['TYPE'] == 3 ),'Unit Type'] = 'Attached'

    
    data.loc[(data['AIRSYS'] == 2) & (data['year'] < 2015), 'AIRSYS'] = 0.0
    data.loc[(data['AIRSYS'] == 12) & (data['year'] >= 2015), 'AIRSYS'] = 0.0
    data.loc[(data['AIRSYS'] != 12) & (data['year'] >= 2015), 'AIRSYS'] = 1.0
    
    #for purposes of matching 
    #data = data[data.year < 2015]
               
            
    for i in data.year.unique():
        if i >=2015 and i<=2017:
            data.loc[data.year==i,'BATHS'] = data.loc[data.year==i,'BATHS'].replace({7: 0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 3:2,4:3,5:3,6:4 })
        else:
            data.loc[data.year==i,'BATHS'] = data.loc[data.year==i,'BATHS'].replace({5:4, 6:4, 7: 4, 8:4, 9:4, 10:4, 11:4, 12:4, 13:4})
    

    data = data.sort_values(['MSA', 'year'], axis = 0)
    
    all_data = data
    
    
    
    data = data.loc[data['Unit Type'].isin(['Detached', 'Attached'])]
    data.loc[data['Unit Type'] == 'Detached','Unit Type'] = 1.0
    data.loc[data['Unit Type'] == 'Attached','Unit Type'] = 0.0
    
    hedonic_data  = pd.get_dummies(data, columns = ['year', 'MSA', 'BUILT'])
    all_data= pd.get_dummies(all_data, columns = ['year', 'MSA', 'BUILT'])
    
    """convert the airsys variable to a dummy variable which is binary"""
    
    hedonic_data['intercept'] = 1.0
    all_data['intercept'] = 1.0
    
    x_var = hedonic_data.columns.tolist()
    x_var.remove('log rent')
    x_var.remove('TENURE')
    x_var.remove('CONDO')
    x_var.remove('NUNIT2')
    x_var.remove('TYPE')
    x_var.remove('year_1985')
    x_var.remove('MSA_360.0')
    x_var.remove('BUILT_1910')
    
    
    owned_data = hedonic_data[hedonic_data['log rent'].isnull()]
    hedonic_data = hedonic_data[hedonic_data['log rent'] != 0.0]
    
    model = regression.linear_model.OLS(hedonic_data['log rent'], hedonic_data[x_var])
    results = model.fit()
    print(results.summary())


    #aggregate the rent-to-price ratios with nonparameteric weights
    #predict the rent on the owned-occupied
    
    owned_data.loc[:,'log rent'] = (owned_data * results.params).sum(axis = 1)
    
    df = pd.concat([owned_data, hedonic_data], axis = 0)
    df['rent'] = np.exp(df['log rent'])
        
