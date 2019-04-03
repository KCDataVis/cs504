#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 11:18:22 2019

@author: zackkingbackup
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import datetime

def plotTicker(df, ticker, start_time='beginning',days='all', title="none"):
    '''
        Input start_time in years
        days is the duration of the time period
    '''
    if title == 'none': title = ticker
    tic = df[df['Name'] == ticker]
    if start_time == 'beginning' and days == 'all':
        pass
    else:
        tic = tic[tic['FloatYear'] > start_time]
        tic = tic.iloc[:days]
    plt.title(title)
    plt.plot(tic['FloatYear'],tic['close'],'-',label=ticker)
    plt.xlabel("Years")
    plt.ylabel("Stock Closing Price")
    
def plotMatch(ya, pred, yeval, xvalst, xvalsp, title='def',xy='none',
              savefile='none'):
    '''
        plotMatch(ya, pred, yeval, xvalst, xvalsp, title='def',xy='none',
        savefile='none')
            Visualize the forecast compared with the actual price trend
            
            Parameters
            ----------
            ya : numpy.array
                closing prices over start_date->start_date + pattern_length
                days of the target stock
            pred : numpy.array
                pred_length forecast of upcoming daily closing prices
            yeval : numpy.array
                pred_length historical record of closing prices of target stock
            xvalst : numpy.array
                numpy.arange(pattern_length)
            xvalsp : numpy.array
                numpy.arange(pattern_length, pattern_length+pred_length)
            title : str ; default "Forecasting using Chart Patterns"
                title of visualization
            xy : tuple, list, numpy.array, iterable ; default (centered in x
            and at the height of mean(ya))
                coordinates of annotation of squared error of the forecast
                to the real price history
            savefile : str ; default None
                file to save the visualization off to
            
            Returns
            -------
            None
    '''
            
    
    plt.plot(xvalst, ya, label='Actual',color='b')
    plt.plot(xvalsp, pred,label="Model Forecast",color='g')
    if len(yeval) == 0:
        pass
    else:
        plt.plot(xvalsp, yeval, label='Actual)',color='r')
    plt.legend(loc='best')
    plt.xlabel("Days")
    plt.ylabel("Stock Closing Price")
    if title == 'def':
        plt.title("Forecasting using Chart Patterns")
    else:
        plt.title(title)
    if xy == 'none':
        xav = (len(xvalst)+len(xvalsp))/2
        yav = (np.mean(ya))
    else:
        xav = xy[0]
        yav = xy[1]
    if len(yeval) > 0:
        LSE_forecast= getLSE(yeval,pred)
        plt.annotate("LSE = %8.1f"%(LSE_forecast),[xav,yav])
    if savefile != "none":
        plt.savefig(savefile)

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

def inverse(LSE):
    '''
        provides weights for stock charts inversely proportional
        to their squared errors after scaling compared to the target stock
    '''
    weights = 1./np.array(LSE)
    sumW = sum(weights)
    return weights, sumW

def pick_comparables(df_pats, n_comp=20):
    
    # crude method of picking similar stocks by selecting the n_comp
    # smallest squared errors when scaled to fit the target stock
    temp = df_pats.copy()
    temp.sort_values(by='LSE',inplace=True)
    return temp.index[:n_comp]

def getDate(year, month, day):
    return "%4d"%(year)+"-%02d"%(month)+"-%02d"%(day)

def match(df, pattern_length, pred_length, target, start_date, n_comp=20, 
          weighting=inverse, n_rand_start=50, random_seed='none',
          pick_comparables=pick_comparables):
    '''
       match(df, pattern_length, pred_length, target, start_date, n_comp=20,
       weighting=inverse, n_rand_start=50) 
           Output forecast of pred_length days for stock ticker equal to target
           
           Parameters
           ----------
           df : pandas.DataFrame
               This dataframe must have the same column names as
               all_stocks_5yr_pctdiff_vol.csv or an error will be thrown
           pattern_length : int (days)
               Number of days past start_date to analyze; the length of
               the chart pattern the technician has identified and wants
               to do quantitative historical analyses of
           pred_length : int (days)
               How long should the forecast be? 1 day? 10 days? A year?
           target : str
               Ticker of the stock for which the technician has identified
               a chart pattern. If the ticker is not included in the dataset
               passed in with df, an error will be thrown
           start_date : datetime.datetime
               Date the chart pattern to be analyzed begins. Initialize a
               datetime.datetime by: date = datetime.datetime(2019,3,30)
               to indicate 3/30/2019
           n_comp : int , default 20
               Number of comparable stock charts to use in the final 
               output forecast
           weighting : function(LSE) returning weights, sum(weights) ;
           default inverse
               Weighting function that takes the squared errors of the n_comp
               scaled stock charts with the target stock's chart over the
               pattern dates and ouputs the weights for each n_comp chart
               along with the sum of the weights for normalization
           n_rand_start : int , default 50
               Number of possible comparable charts to generate from each
               input stock chart. The starting dates are drawn from a 
               uniform random distribution. The stock chart over the 
               length of the pattern from the start date is considered
               as a possible n_comp chart. Theoretically you increase this
               number to increase the liklihood you find a few great matches
               for your particular chart pattern from across your data. Setting
               this number too low all but eliminates the chance your're going
               to find any pattern close to what you're looking for. I actually
               don't see any correlation between n_rand_start and perforamnce
               though - could just be weird examples I've tried so far.
           random_seed : int , default none
               seed for random number generator that picks starting points
               along stock curves to draw samples of length pattern_length
               for consideration in the forecast of the target stock
           pick_comparables : function , default function(pandas.DataFrame
           containing columns 'LSE','Sls')
               Function that picks which similar stock charts to use
               to make the forecast of the target stock
               
           Returns
           -------
           ya, PRED, yEVAL, xvalst, xvalsp : all 5 outputs are 1-D numpy.array 
           instances. PRED and yEVAL are of length pred_length; PRED is the
           forecast, yEVAL is what your stock actually did over the prediction
           timeframe. xvalst, xvalsp are arrays of placeholders for days
           for plotting purposes. xvalst is length pattern_length, xvalsp
           is of length pred_length. ya is the actual prices over 
           pattern_length. Plot with xvalst
    '''
    
    pattern_time = pattern_length
    pred_time = pred_length
    
    #target_start_date = start_date
    year, month, day = start_date.year, start_date.month, start_date.day
    mydate = getDate(year, month, day)
    
    # did the user ask for a weekend or federal holiday? Lets make sure this
    # date is in the dataset
    notify = False
    while len(df[(df['Name'] == target) & (df['date'] == mydate)]) == 0:
        notify = True
        day += 1
        if day > 31:
            day = 22
        mydate = getDate(year, month, day)
             
    if notify:
        print("Your begin date fell on a day in which the exchange wasn't open or your\
 stock wasn't being publicly traded. Changed date to: %4d-%02d-%02d"%(year,month,day))
    DTs = []
    DTFuture = []
    LSEs = [] 
    
    tardf = pd.DataFrame()
    testdf = pd.DataFrame()
    temp = df[df['Name'] == target].sort_values(by='FloatYear')
    idx = int(temp[(temp['date'] == mydate) & (temp['Name'] == target)].index[0])
    target_start_date = np.where(temp.index == idx)[0][0]
    tardf = temp.iloc[target_start_date:target_start_date+pattern_time]
    
    # look out for the case where we're making a future prediction - there's
    # incomplete data available for what actually happened because it hasn't happened yet!
    try:
        testdf = temp.iloc[target_start_date+pattern_time:target_start_date+pattern_time+pred_time]
    except:
        testdf = temp.iloc[target_start_date+pattern_time:]
        # if len(testdf) == 0: do something smart
    
    if random_seed == 'none':
        pass
    else:
        np.random.seed(random_seed)
    
    for stock in df.Name.unique():
        # don't process your target stock
        if stock == target: 
            continue
        # dfss is orig DataFrame by Stock and Sorted by time
        # try/except block to gaurd against there not being a particular
        # ticker in the dataset
        try:
            dfss = df[df['Name'] == stock].sort_values(by='FloatYear')
        except:
            print ("Problem with: "+str(stock))
            continue
        
        # catch the case where the stock has recently become publicly
        # tradeable and it doesn't have enought data for us to 
        # actually use 
        # (ex: need 2 years of data, company made public 6 months ago)
        if len(dfss) - pattern_time - pred_time < 1: continue
        
        cnt = n_rand_start
        while cnt > 0:
            # pick a random starting point
            start = np.random.randint(0,len(dfss) - pattern_time - pred_time)
            dfT = dfss.iloc[start:start+pattern_time]
            dfF = dfss.iloc[start+pattern_time:start+pred_time+pattern_time]
            
            DTs.append(dataTime(dfT))
            DTFuture.append(dataTime(dfF))
    
            cnt -= 1
    
    # dataTime class instance initialized with a dataframe containing
    # only rows from the input df corresponding to the target stock
    # during which the chart pattern was observed (length pattern_length)
    Actual_DT = dataTime(tardf)  
    # closing prices throughout the chart pattern; real values
    ya = Actual_DT.closes
    
    # Create lists to hold dataTime instances corresponding to 
    # scaled stock charts
    #
    # dataTime holds information about a stock over a given time period
    # so instances of dataTime are used to hold onto pricing over random
    # time intervals        
    DTsFit = []
    DTFutureFit = []
    Slss = []
    
    for i in range(len(DTs)):
        
        Sls = computeSls(Actual_DT, DTs[i])
        Slss.append(Sls)
        DTsFit.append(Sls*DTs[i].closes)
        DTFutureFit.append(Sls*DTFuture[i].closes)
        
        LSEs.append(getLSE(ya, DTsFit[-1]))
    
    # these LSEs are used for the weights we'll apply to the DTFuture
    # stock time series while averaging them to make a forecast
    
    # choose a subset of patterns to use for weighted average
    # put a function in here to do this; 
    # it must output a dataframe
    # containing the LSE values + any values to be used in weighting
    # later on. This dataframe has to have indices that trace back to
    # indices in the DTsFutureFit/LSEs/DTsFit lists
    df_pats = pd.DataFrame({'LSE' : LSEs, 'Sls' : Slss, 'DTsFit' : DTsFit})
   
    # use the passed in pick_comparables function to determine the
    # indices into the lists containing our stock charts -
    # nominally these are just the charts with the n_comp lowest 
    # squared errors
    idxs = pick_comparables(df_pats, n_comp)
    LSEw = df_pats.loc[idxs]['LSE']
    
    # apply weights via the weighting function passed in
    weights, sumW = weighting(LSE=LSEw)
    
    prediction = []
    for i in range(pred_time):
        # for each time step, predict closing price
        pred_close = 0
        
        # iterate through each comparable
        cnt1 = 0
        for j in idxs:
            # compute a weighted average for each 'day'
            close = DTFutureFit[j][i]
            pred_close += close*weights[cnt1]/sumW
            cnt1 += 1
        # append the day's prediction to the prediction list
        prediction.append(pred_close)
    try:
        Eval = dataTime(testdf)
        yEVAL = Eval.closes
    except:
        yEVAL = np.array([])
    
    xvalst = np.arange(pattern_time)
    xvalsp = np.arange(pattern_time,pattern_time+pred_time)
    PRED = np.array(prediction)
    ya = np.array(ya)
    
    return ya, PRED, yEVAL, xvalst, xvalsp

    
    
    
    
    






