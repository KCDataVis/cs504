#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:17:04 2019

@author: zackkingbackup
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def plotTicker(df, ticker, start_time='beginning',days='all'):
    '''
        Input start_time in years
        days is the duration of the time period
    '''
    tic = df[df['Name'] == ticker]
    if start_time == 'beginning' and days == 'all':
        pass
    else:
        tic = tic[tic['FloatYear'] > start_time]
        tic = tic.iloc[:days]
    plt.plot(tic['FloatYear'],tic['close'],'-',label=ticker)
    plt.xlabel("Years")
    plt.ylabel("Stock Closing Price")

class dataTime(object):
    
    startDates = []
    
    def __init__ (self, df):
        
        self.t = len(df) # time period in days
        self.startDate = df.iloc[0]['date']
        dataTime.startDates.append(df.iloc[0]['FloatYear'])
        self.endDate = df.iloc[-1]['date']
        self.name = df.iloc[0]['Name']+str(self.startDate)
        self.closes = np.array(df['close'])
        self.df = df.copy()
        
def computeSls(dataTime1, dataTime2):
    '''
        dataTime1 is your target
        dataTIme2 is your 'prediction'
    '''
    ya = dataTime1.closes
    yp = dataTime2.closes
    
    '''
        Sls = sum, i from 0->len(ya)-1 of ya_i*yp_i / 
              sum, i from 0->len(ya)-1 of yp_i**2
        
        no you can't cancel these terms because they are in 
        distinct summations.
    '''
    return sum(ya*yp)/sum(yp**2)

def getLSE(ya,yS):
    '''
        pass in two numpy arrays holding the target stock
        chart (ya) and the scaled predictor stock chart (yS).
        Get the least squares error (LSE) out.
    '''
    return sum((ya - yS)**2)

df = pd.read_csv("all_stocks_5yr_pctdiff_vol.csv")

''' 
    lets get ourselves a collection of time series data
    of stocks over an arbitrary time period ex: 10 days
    
    pattern_time holds the length of the time period in days
    pred_time holds the length of time we want to forecast for
'''

# setting up a classic head and shoulders case;
# when there are three consecutive peaks, angled down, up
# or neutral, it is a time to short because the price is about
# to descend. We will use this example to try to teach the 
# machine to recognize the head and shoulders pattern
# returning other examples, using them we will see if our
# forecast is a gloomy one as common knowledge would dictate.t
pattern_time = 330
pred_time = 70
# for each stock, pick many uniformly random start points between
# t(min) and t(max) - t and grab the next t-1 stock values
# also grab the data points in the future we want to use
# for forecasting

# place them in a list of dataTime instances
# also define a list for scaled Least Squares errors to use later in a 
# tbd weighting function (defaulted it to inverse of the error)
DTs = []
DTFuture = []
LSEs = []

# define a stock we're predicting for
target = 'ABC'
target_start_date = 820 # let this simply be an index into itself for now
tardf = pd.DataFrame()
testdf = pd.DataFrame()
np.random.seed(7)

for stock in df.Name.unique():
    # don't process your target stock
    if stock == target: 
        temp = df[df['Name'] == stock].sort_values(by='FloatYear')
        tardf = temp.iloc[target_start_date:target_start_date+pattern_time]
        testdf = temp.iloc[target_start_date+pattern_time:target_start_date+pattern_time+pred_time]
        continue
    # dfss is orig DataFrame by Stock and Sorted by time
    dfss = df[df['Name'] == stock].sort_values(by='FloatYear')
    
    # catch the case where the stock has recently become publicly
    # tradeable and it doesn't have enought data for us to 
    # actually use 
    # (ex: need 2 years of data, company made public 6 months ago)
    if len(dfss) - pattern_time - pred_time < 1: continue
    
    cnt = 50
    while cnt > 0:
        # pick a random starting point
        start = np.random.randint(0,len(dfss) - pattern_time - pred_time)
        dfT = dfss.iloc[start:start+pattern_time]
        dfF = dfss.iloc[start+pattern_time:start+pred_time+pattern_time]
        
        DTs.append(dataTime(dfT))
        DTFuture.append(dataTime(dfF))

        cnt -= 1

# plt.hist(dataTime.startDates, bins=50)

# now we have 10,080 price/time=t templates (504*20)
# scale them to minimize the least squared error between
# the template and the target
Adt = dataTime(tardf)    
ya = Adt.closes
        
DTsFit = []
DTFutureFit = []

for i in range(len(DTs)):
    
    Sls = computeSls(Adt, DTs[i])
    DTsFit.append(Sls*DTs[i].closes)
    DTFutureFit.append(Sls*DTFuture[i].closes)
    
    LSEs.append(getLSE(ya, DTsFit[-1]))
    #LSEs.append(getLSE(Adt.closes, DTs[i].closes))

# these LSEs are used for the weights we'll apply to the DTFuture
# stock time series while averaging them to make a forecast

# choose a subset of patterns to use for weighted average
df_pats = pd.DataFrame({'LSE' : LSEs})
df_pats.sort_values(by='LSE',inplace=True)

# how many templates should I use?
n_comp = 20

# errors for use in weighting
LSEw = df_pats.iloc[:n_comp]['LSE']

# make the weight computation a function, try a few different functions
# and note how the resulting model performance differs
weights = 1./np.array(LSEw)
sumW = sum(weights)

prediction = []
for i in range(pred_time):
    # for each time step, predict closing price
    pred_close = 0
    
    # iterate through each comparable
    # not that DTFuture contains EVERY other pattern;
    # perhaps introduce a function to narrow down this list
    #for j in range(len(DTFuture)):
    cnt1 = 0
    for j in df_pats.index[:n_comp]:
        # compute a weighted average for each 'day'
        close = DTFutureFit[j][i]
        pred_close += close*weights[cnt1]/sumW
        cnt1 += 1
    # append the day's prediction to the prediction list
    prediction.append(pred_close)

Eval = dataTime(testdf)
yEVAL = Eval.closes

xvalst = np.arange(pattern_time)
xvalsp = np.arange(pattern_time,pattern_time+pred_time)

plt.plot(xvalst, ya, label='Actual('+str(Adt.name)+")",color='b')
plt.plot(xvalsp, prediction,label="Model Forecast",color='g')
plt.plot(xvalsp, yEVAL, label='Actual('+str(Eval.name)+")",color='r')
plt.legend(loc='best')
plt.xlabel("Days")
plt.ylabel("Stock Closing Price")
plt.title("Forecasting using Chart Patterns")
xav = (pattern_time+pred_time)/2
yav = (np.mean(yEVAL))
LSE_forecast= getLSE(yEVAL,prediction)
plt.annotate("LSE = %8.1f"%(LSE_forecast),[xav,yav])
plt.savefig("Images/LSforecastHS1.png")

plt.figure()
for j in df_pats.index[:n_comp]:
    plt.plot(xvalst, DTsFit[j],'-',label=str(DTs[j].name),markersize=1)
    plt.plot(xvalsp, DTFutureFit[j], '-',label=str(DTFuture[j].name),markersize=1)

plt.plot(xvalst, ya, label='Actual('+str(Adt.name)+")",color='b',markersize=3)
plt.plot(xvalsp, yEVAL, label='Actual('+str(Eval.name)+")",color='r')

plt.xlabel("Days")
plt.ylabel("Stock Closing Price")
plt.title("Similar Patterns to "+str(Adt.name)+" (Head and Shoulders)")
plt.savefig("Images/SimilarPatternsHS.png")

plt.figure()
plotTicker(df,'ABC')
plt.title("ABC")
plt.savefig("Images/ABCchart.png")
plt.show()

'''   
# demo
dt1 = DTs[0]
dt2 = DTs[500]
ya = dt1.closes
yp = dt2.closes
yS = yp*computeSls(dt1,dt2)

xvals = np.arange(t)

plt.plot(xvals, ya, '-',label='Actual('+str(dt1.name)+")",color='b')
plt.plot(xvals, yp, '-',label='Prediction('+str(dt2.name)+")",color='g')
plt.plot(xvals, yS, '-',label='Scaled('+str(dt2.name)+")",color='r')
plt.legend(loc='best')
plt.xlabel("Days")
plt.ylabel("Stock Closing Price")
plt.title("Least Squares Fitting of Stock Charts")
xav = t/2
yav = (max(yS) + max(yp)) / 2
plt.annotate("LSE = %8.1f"%(getLSE(ya,yS)),[xav,yav])
#plt.savefig("Images/LSexample1.png")
'''











