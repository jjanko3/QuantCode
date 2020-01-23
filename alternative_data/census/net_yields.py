# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:59:16 2019

@author: jjanko
"""


import pandas as pd
import os


if __name__ == "__main__":
    
    #get the vacancy rates
    rental_data = pd.read_hdf(os.path.join(r'T:\PhD Students','rental_data.h5'))
    
    
    vacancy = {}
    for i in rental_data.CITY.unique():
        vacancy[i] = {}
        for j in rental_data.year.unique():
            try:
                vacancy[i][j] = float(rental_data.loc[((rental_data['CITY']==i)) & (rental_data['year']==j)]['VACANCY'][rental_data.loc[((rental_data['CITY']==i)) & (rental_data['year']==j)]['VACANCY'].isin([1,2,4])].count()) / float(rental_data.loc[((rental_data['CITY']==i)) & (rental_data['year']==j)]['VACANCY'].count())
            except ZeroDivisionError :
                vacancy[i][j] = 0.0
            
        
    #zero indicates that there was no data for that year    
    vacancy_rates = pd.DataFrame(vacancy).transpose()
    print(vacancy_rates)