# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 07:16:14 2019

@author: jjanko
"""


import pandas as pd
import os
import numpy as np
from statsmodels import regression


def change_airsys_data(data):
    
    data.loc[(data['AIRSYS'] == 2) & (data['year'] < 2015), 'AIRSYS'] = 0.0
    data.loc[(data['AIRSYS'] == 12) & (data['year'] >= 2015), 'AIRSYS'] = 0.0
    data.loc[(data['AIRSYS'] != 12) & (data['year'] >= 2015), 'AIRSYS'] = 1.0
    
    return data

def change_bath_data(data):
    for i in data.year.unique():
        if i >=2015 and i<=2017:
            data.loc[data.year==i,'BATHS'] = data.loc[data.year==i,'BATHS'].replace({7: 0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 3:2,4:3,5:3,6:4 })
        else:
            data.loc[data.year==i,'BATHS'] = data.loc[data.year==i,'BATHS'].replace({5:4, 6:4, 7: 4, 8:4, 9:4, 10:4, 11:4, 12:4, 13:4})
    return data

def change_unit_type_data(data):
    data['Unit Type'] = 'None' 
    data.loc[data['CONDO'] == 1, 'Unit Type'] = 'Condo'

    data.loc[(data['year'] < 2015) & (data['NUNIT2'] == 1 ),'Unit Type'] = 'Detached'
    data.loc[(data['year'] < 2015) & (data['NUNIT2'] == 2 ),'Unit Type'] = 'Attached'
    
    
    data.loc[(data['year'] >= 2015) & (data['TYPE'] == 2 ),'Unit Type'] = 'Detached'
    data.loc[(data['year'] >= 2015) & (data['TYPE'] == 3 ),'Unit Type'] = 'Attached'
    
    
    return data
    

def change_build_data(data):
    
    for i in data.year.unique():
        data.loc[data.year==i,'BUILT'] = data.loc[data.year==i,'BUILT'].replace({80: 1980,
                81:1981,82:1982,83:1983,84:1984,85:1985,86:1986,87:1987,88:1988,89:1989,
                90:1990,91:1991,92:1992,93:1993,94:1994,95:1995,96:1996,97:1997,98:1998,
                99:1999})
             
    for i in data.year.unique():
        if i >=1985 and i<=1995:
            data.loc[data.year==i,'BUILT'] = data.loc[data.year==i,'BUILT'].replace({9: 1919,
                    8:1920,7:1930,6:1940,5:1950,4:1960,3:1970,2:1970,1:1970})
    
    for i in data.year.unique():
        data.loc[data.year==i,'BUILT']=data.loc[data.year==i,'BUILT'].replace({1919:1910,
             1981:1980,1982:1980, 1983:1980,1984:1980,1986:1985,1987:1985,1988:1985,1989:1985,
             1991:1990,1992:1990, 1993:1990,1994:1990,1996:1995,1997:1995,1998:1995,1999:1995,
             2001:2000,2002:2000, 2003:2000,2004:2000,2006:2005,2007:2005,2008:2005,2009:2005,
             2011:2010,2012:2010, 2013:2010,2014:2010})
    #for i in data.year.unique():
    #    data.loc[data.year==i,'BUILT']=(np.floor(data.loc[data.year==i,'BUILT']/10.0)*10.0).astype(int)
                        
                        
                
        
    for i in data.year.unique():
        if i >=2015 and i<=2017:
            data.loc[data.year==i,'data'] = data.loc[data.year==i,'AIRSYS'].replace({2: 1, 3:1, 4:1, 5:1, 6:1, 7:1, 8:1, 9:1, 10:1, 11:1,12:2})
             
    for i in data.year.unique():
        if i >=2015 and i<=2017:
            data.loc[data.year==i,'BATHS'] = data.loc[data.year==i,'BATHS'].replace({7: 0, 8:0, 9:0, 10:0, 11:0, 12:0, 13:0, 3:2,4:3,5:3,6:4 })
        else:
            data.loc[data.year==i,'BATHS'] = data.loc[data.year==i,'BATHS'].replace({5:4, 6:4, 7: 4, 8:4, 9:4, 10:4, 11:4, 12:4, 13:4})
        
    return data
    

if __name__ == "__main__":
    
    same_smsa = {'35620':5600,'31080':4480,'16980':1600,'19100':1920,'37980':6160,'26420':3360,'47900':8840,'33100':5000,'12060':520,'14460':1120,'41860':7360,'19820':2160,'40140':6780,'38060':6200,'42660':7600,'19740':2080,'38300':6280,'38900':6440,'17140':1640,'17460':1680,'28140':3760,'33340':5080,'32820':4920,'35380':5560,'39580':6640,'33460':5120,'45300':8280,'12580':720,'41700':7240,'29820':4120,'41940':7400,'36420':5880,'40060':6760,'13820':1000,'40380':6840}
    same_smsa_back = {'5600': 35620, '4480': 31080, '1600': 16980, '1920': 19100, '6160': 37980, '3360': 26420, '8840': 47900, '5000': 33100, '520': 12060, '1120': 14460, '7360': 41860, '2160': 19820, '6780': 40140, '6200': 38060, '7600': 42660, '2080': 19740, '6280': 38300, '6440': 38900, '1640': 17140, '1680': 17460, '3760': 28140, '5080': 33340, '4920': 32820, '5560': 35380, '6640': 39580, '5120': 33460, '8280': 45300, '720': 12580, '7240': 41700, '4120': 29820, '7400': 41940, '5880': 36420, '6760': 40060, '1000': 13820, '6840': 40380}
    
    eisfeldt_top_30 = {"6280": "Pittsburgh","7040": "St. Louis","3760": "Kansas City","3360": "Houston","1920": "Dallas","5880": "Oklahoma City","1680": "Cleveland ","6160": "Philadelphia","520": "Atlanta","5120": "Minneapolis","7600": "Seatle","720": "Baltimore","5380": "Nassau-Suffolk","5680": "Virginia Beach","1600": "Chicago","8280": "Tampa","8840": "Washington","5000": "Miami","7360": "San Fransico","1120": "Boston","7320": "San Diego","7400": "San Jose","360": "Anaheim","2160": "Detroit","5640": "Newark","5600": "New York","5775": "Oakland","4480": "Los Angeles","6200": "Phoenix","6780": "Riverside"}

    #pre 2015 smsa values for filtering
    vals = []
    for iterate in eisfeldt_top_30.keys():
        vals.append(float(iterate))
    
    #after 2015 smsa values for filtering
    _vals = []
    for v in vals:
        if str(int(v)) in same_smsa_back.keys():
            _vals.append(same_smsa_back[str(int(v))])

    #directory the data
    filepath = r'T:\PhD Students\ftp_data_census2'
    
    df = pd.DataFrame()
    sample = {}
    #go through each year directory
    for item in os.listdir(filepath):
        if item[0] == 'o':
            year = int(item.split(".")[0][1:])
            data = pd.read_hdf(os.path.join(filepath, item))
            
            #get the top 30 cities according to eisfeldt
            if year >= 2015:
                data = data.loc[data['SMSA'].isin(_vals)]
            else:
                data = data.loc[data['SMSA'].isin(vals)]
            
            #merging all the data into one dataframe
            if len(df.index) == 0:
                df = data.copy()
            else:
                df = pd.concat([df, data.copy()], axis = 0)
                
    
    #clean the data
    #get the log rent, nan values for negatives
    df['log rent'] = np.log(df['RENT'])
    #check for errors in room
    df = df.loc[df['ROOMS'] >= 0]
    #check for errors in household income
    df = df.loc[~df['ZINC2'].isin([-6,-9])]
     
    #define the original dataset just in case you need to look back at it 
    original = df.copy()
    original_sample = df['year'].value_counts()
    
    #make changes to response codes
    df = change_build_data(df)
    df = change_unit_type_data(df)
    df = change_airsys_data(df)
    df = change_bath_data(df)

    

    #remove rent control
    df = df.loc[~((df['RCNTRL'].isin([8,1,2])) & (df['year'] < 1997 ) )]
    df = df.loc[~((df['RCNTRL'].isin([1,-7,-8,-9])) & (df['year'] >= 1997 ) )]
   
    rent_control = df['year'].value_counts()
    print(rent_control - original_sample)
    
    """
    #remove windows
    df = df.loc[~((df['EBAR'].isin([9,1,8])) & (df['year'] < 1997 ) )]
    df = df.loc[~((df['EBAR'].isin([1,-7,-8,-9])) & (df['year'] >= 1997 ) )]
    
    no_bars = df['year'].value_counts()
    print(no_bars - original_sample)
    """
    
    
    

    
    #filter income/house value according to Eisfeldt
    df = df[(df['ZINC2'] / df['VALUE']) <= 2. ] 
    income_value_ratio = df['year'].value_counts()
    print(income_value_ratio - original_sample)
    
    #filter income/rent according to Eisfeldt
    df = df.loc[(df['ZINC2'] /(12. * df['RENT'])) <= 100. ] 
    income_rent_ratio = df['year'].value_counts()
    print(income_rent_ratio - original_sample)
    
    #filter for unit type based on Halket's reccomendation
    df = df.loc[df['Unit Type'].isin(['Detached', 'Attached'])]
    df.loc[df['Unit Type'] == 'Detached','Unit Type'] = 1.0
    df.loc[df['Unit Type'] == 'Attached','Unit Type'] = 0.0
    
    #define a new dataset for regression analysis and price ratio anaylsis
    r_data = df[['BUILT','ROOMS','BATHS','AIRSYS','BEDRMS','Unit Type','SMSA','year', 'log rent', 'VALUE']]
    r_data = r_data.sort_values(['SMSA', 'year'], axis = 0)

    #get the owned single family homes based off null value in rent
    owned_data = r_data[r_data['log rent'].isnull()]
    #clean the data for home values greater than zero for ratio analysis
    owned_data = owned_data[owned_data['VALUE'] > 0]
    
    #get all rents greater than zero, assume error in data if log rent less than zero
    hedonic_data = r_data[r_data['log rent'] > 0.0]
    
    #get the dummies for MSA, years, year built
    hedonic_data  = pd.get_dummies(hedonic_data, columns = ['year', 'SMSA', 'BUILT'])
    
    #create an intercept column
    hedonic_data['intercept'] = 1.0
    
    #get the indepent variables for hedonic regression
    x_var = hedonic_data.columns.tolist()
    #remove variables we do not need and get rid of one dummy to prevent multi-collinearity
    x_var.remove('year_1985')
    x_var.remove('SMSA_360.0')
    x_var.remove('BUILT_1910')
    x_var.remove('log rent')
    x_var.remove('VALUE')
    
    

    #run the regression y on x
    model = regression.linear_model.OLS(hedonic_data['log rent'], hedonic_data[x_var])
    #fit the model
    results = model.fit()
    #print a summary of the results
    print(results.summary())
 
    