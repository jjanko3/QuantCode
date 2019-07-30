#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:23:28 2018

@author: jj
"""

import pandas as pd
from iexfinance import get_historical_data
import datetime
import statsmodels.api as sm


def getReturnDf(stock_list, start, end):
    prices = pd.DataFrame()
    
    for stock in stock_list:
        data = get_historical_data(stock, start=start, end=end, output_format='pandas')['close']
        data = pd.DataFrame(data)
        data.columns = [stock]
        data.index = pd.to_datetime(data.index)
        if prices.empty:
            prices = data.copy()
        else:
            prices = pd.concat([prices, data], axis  = 1)

    returns = prices.pct_change().dropna(axis = 0)

    return returns


if __name__ == '__main__':
    
    #quandl.ApiConfig.api_key = 'cwAFdAtv1-nYDZUFyFeP'
    n_years = 3 * 365

    end = datetime.date.today()
    start = end - datetime.timedelta(days=n_years)  

    stock_symbols = ['AAPL','FB','MSFT']
    
    #SPY is the market
    #SHY is short term treasuries to reflect rf
    #VV is large cap 
    #VB is small cap
    #VTV is value
    #VUG is growth
    
    factor_model = ['SPY','SHY','VV','VB','VTV','VUG']


    returns = getReturnDf(stock_symbols, start, end).resample('M').sum()
    factor = getReturnDf(factor_model, start, end).resample('M').sum()
    
    rf = factor['SHY']
    mkt = factor['SPY'] - factor['SHY']
    mkt.columns = ['market']
    mkt.name  = 'mkt'
    smb = factor['VB'] - factor['VV']
    smb.columns = ['smb']
    smb.name = 'smb'
    hml = factor['VTV'] - factor['VUG']
    hml.columns = ['hml']
    hml.name = 'hml'
    alpha = factor['SPY']
    alpha.columns = ['alpha']
    alpha.name = 'alpha'
    alpha[alpha != 1.0] = 1.0
    
    print(returns.head(20))
    print(factor.head(20))
    
    
    for stock in returns:
        
        fact = pd.concat([ mkt, smb, hml,alpha], axis = 1)
        model = sm.OLS(returns[stock], fact)
        results = model.fit()
        print('In Sample Results ' + stock)
        print(results.summary())
        print(' ')
        print('Out of Sample Results ' + stock)
        model = sm.OLS(returns[stock][1:], fact.shift(1)[1:])
        os_results = model.fit()
        print('In Sample Results ' + stock)
        print(os_results.summary())
        print(' ')
        
        