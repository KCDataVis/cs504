#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:49:56 2019

@author: zackkingbackup
"""

import MatchFunction as mf
import datetime
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt

start_time = time.time()
'''
Decent enough Head and Shoulders example:

target = 'MMM'
start_date = datetime.datetime(2014,1,20)
pattern_length = 500
prediction_length = 100

Change the variables below, however, make a note somewhere if you
stumble upon another chart pattern that we need to be looking at!

'''
target = 'MMM'
start_date = datetime.datetime(2014,4,10)
pattern_length = 500
prediction_length = 100

# read in the data
# crucial that you use the csv I uploaded to BB File Exchange (too big for 
# Github) titled "all_stocks_5yr_pctdiff_vol.csv"
# it has processed date columns that make it easier for the match
# function to pick off the appropriate rows of the csv for analysis
df = pd.read_csv("../all_stocks_5yr_pctdiff_vol.csv")

mt1 = time.time()
# use the chart pattern matching algorithm to output a forecast
# I've got a few options in here... try help(mf.match) to see
ya, pred, yeval, xvalst, xvalsp, df_pats = mf.match(df=df,
                                                    pattern_length=pattern_length,
                                                    pred_length=prediction_length,
                                                    target=target,
                                                    start_date=start_date,
                                                    n_rand_start=20,
                                                    random_seed=10)
mt2 = time.time()
print ("MatchFunction.match ran in %6.2f"%(mt2 - mt1)+" seconds")
# Score the forecast
# We'll keep track of these in the future as we grid search over
# algorithm hyperparameters later to find optimal performance
SE_forecast = mf.getLSE(pred,yeval)

# Visualize the forecast and historical data
# there's documentation if you type help(mf.PlotMatch)
mf.plotMatch(ya, pred, yeval, xvalst, xvalsp,title=target,xy=[50,58])

end_time = time.time()
print ("MatchAlgoExample.py ran in %6.2f"%(end_time - start_time)+" seconds")