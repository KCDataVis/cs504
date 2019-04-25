#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 17:17:14 2019

@author: zackkingbackup
"""

import pandas as pd

import visualize as v

# read in the data
# crucial that you use the csv I uploaded to BB File Exchange (too big for 
# Github) titled "all_stocks_5yr_pctdiff_vol.csv"
# it has processed date columns that make it easier for the match
# function to pick off the appropriate rows of the csv for analysis
df = pd.read_csv("../all_stocks_5yr_pctdiff_vol.csv")
image_loc='../Images'

# set up lists of parameters describing all the chart patterns we'd like
# to visualize
tickers = ['DGX',
           'ED',
           'DAL']
pat_types = ['Double Bottoms ~6 months',
             'Cup and Handle ~1 month',
             'Triangles >12 months']
start_dates = [2015.7,
               2016.9,
               2016.4]
end_dates = [2016.24,
             2016.975,
             2016.73]
pred_lengths = [100,
                20,
                50]
for idx in range(len(tickers)):
    target = tickers[idx]
    pattern_type = pat_types[idx]
    start_date = v.getdatewithfloat(df, name=tickers[idx], floatyear=start_dates[idx])
    end_date = v.getdatewithfloat(df, name=tickers[idx], floatyear=end_dates[idx])
    pattern_length = v.getpatternlength(df, tickers[idx], start_date, end_date)
    prediction_length = pred_lengths[idx]
    
    v.plotvariants(df, target, pattern_type, image_loc, pattern_length,
                     prediction_length, start_date, end_date)
