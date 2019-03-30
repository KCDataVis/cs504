#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:44:42 2019

@author: zackkingbackup
"""

import pandas as pd
from datetime import datetime

df = pd.read_csv("all_stocks_5yr.csv")

''' This file looks to be the product of getSandP.py
    which aims to download S&P 500 stock data from the last
    five years. Lets check how many tickers are in the data:
'''
print ("Number of unique tickers: %4d"%(len(df.Name.unique())))

''' >> Number of unique tickers:  505
    Where did we get 5 extra stocks?
    Actually the S&P 500 contains 505 different stocks.
    
    https://www.fool.com/investing/2018/07/10/7-fascinating-facts-about-the-broad-based-sp-500.aspx
    
    What we learn from the fool is that the S&P 500 tracks
    only 500 companies and that the extra stocks are from S&P 500
    using multiple stocks from some of the companies that offer
    more than one class of stock. We might want to remember that
    later on for normalizing the influence of each company - 
    company x shouldn't have twice the influence on our results
    as company y only because x offers more stock classes than y. 
    It might take a little effort to figure out which companies
    appear twice in our data as the tickers will be different.
'''

def getGregorianOrdinal(date):
    month, day, year = getMonthDayYear(date)
    date = datetime(year,month,day)
    return date.toordinal()

def getFloatYear(date):
    month, day, year = getMonthDayYear(date)
    # figure out what to add to year
    # perhaps how many total days?
    month_len = {0 : 0,
                 1 : 31,
                 2 : 28,
                 3 : 31,
                 4 : 30,
                 5 : 31,
                 6 : 30,
                 7 : 31,
                 8 : 31,
                 9 : 30,
                 10: 31,
                 11: 30,
                 12: 31}
    
    # correct the month lengths in the event of a leap year
    ndays = 365 # number of days in a year
    if isLeapYear(year) > 0: 
        month_len[2] = 29
        ndays = 366
   
    # the way this works is set days to the number
    # of days completed in the current month. For March 5, days=4.
    # Add to days the number of days completed in previous months:
    # again if March 5, then 31 + 28 = 59; days += 59
    days = day
    for m in range(month): days += month_len[m]
    
    return year + float(days/ndays)
    
def isLeapYear(year):
    '''
        Leap years actually don't occur every four years
        but its happened to be the case since 1900. 2000 was a leap
        year but 2100 will not be. The years exempted from the every
        four years rule of thumb are those which are divisible by 100
        but not by 400.
    '''
    if year % 4 == 0:
        return 1
    else:
        return 0
    
def getMonthDayYear(date):
    date_string = date.split('-')
    year = int(date_string[0])
    month = int(date_string[1])
    day = int(date_string[2])
    return month, day, year

df['OrdinalDate'] = df.apply(lambda x: getGregorianOrdinal(x['date']), axis=1)
df['FloatYear'] = df.apply(lambda x: getFloatYear(x['date']), axis=1)

''' 
    Create a plot of one stock over the past 5 years
'''
import matplotlib.pyplot as plt
aal = df[df['Name'] == 'AAL']
plt.plot(aal['FloatYear'],aal['close'],'-',markersize=1,label='AAL')
aap = df[df['Name'] == 'AAP']
plt.plot(aal['FloatYear'],aap['close'],'-',markersize=1,label='AAP')

abc = df[df['Name'] == 'ABC']
plt.plot(aal['FloatYear'],abc['close'],'-',markersize=1,label='ABC')


plt.xlabel("Year")
plt.ylabel("Dollars")
plt.title("Closing Stock Prices 2013-2018")
plt.legend(loc='best')
#plt.savefig('Images/StockPlot_032519_AAL_AAP_ABC.png')
plt.show()


'''
    IDEA: Is it possible to
           1. Look at the 'shape' of a particular's stock's
              price/time chart.
           2. Use other stocks' that have had a similar shape
              over an equal time period in the past to predict
                  a. the target stock's future price movement
                  b. the volume traded in the coming <time unit> 
'''

'''
    More feature engineering:
        pct_diff_daily may take values within [-1,inf.) as a stock can't lose
        more than 100% of its value in a day. The percent difference is left
        as a decimal (.06 == 6%).
        
        volatility_daily is a crude metric I arbitrarily invented to quantify
        how volatile a stock was on a given day relative to its average price. 
        Mathematically it is:
            (daily high - daily low) / avg(open, close prices)
'''

df['pct_diff_daily'] = df.apply(lambda x: (x['close'] - x['open'])/x['open'], axis=1)
df['volatility_daily'] = df.apply(lambda x: (x['high'] - x['low'])/(0.5*(x['open']+x['close'])),axis=1)
df.to_csv("all_stocks_5yr_pctdiff_vol.csv",index=False)
'''
    57 / 619,040 stock records had a volatility_daily over 20%
'''
df20p = df[df['volatility_daily'] > .2]

'''
    My volatility metric has a weakness - it is very oversensitive to
    cheap stocks that move less than a dollar; these are a fair number
    of the 57 outliers
'''