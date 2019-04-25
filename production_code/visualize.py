#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:49:56 2019

@author: zackkingbackup
"""

import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os

import MatchFunction as mf

def getdatewithfloat(df, name, floatyear):
    df_temp = df[(df['Name'] == name) & (df['FloatYear'] >= floatyear)]
    date = df_temp.iloc[0].date
    return todatetime(date)

def todatetime(date, sep='-'):
    sldate = date.split(sep)
    y = int(sldate[0])
    m = int(sldate[1])
    d = int(sldate[2].strip())
    return datetime.datetime(y,m,d)

def getpatternlength(df, name, start_date, end_date):
    df_temp = df[df['Name'] == name]
    df_temp['datetime_date'] = df_temp.apply(lambda x: todatetime(x['date']),
           axis=1)
    return len(df_temp[(df_temp['datetime_date'] >= start_date) &
                       (df_temp['datetime_date'] <= end_date)])


def plotvariants(df, target, pattern_type, image_loc, pattern_length,
                 prediction_length, start_date, end_date):


    target_sector = df[df['Name'] == target].iloc[0].sector
    
    # define a DataFrame of stocks within the same sector as the target stock
    dfs = df[df.sector == target_sector]
    
    ##############################################################
    # Lets proceed through 4 model variants for the forecast 
    # of the chart pattern defined by the above target, start_date,
    # end_date
    #
    # 1. All sectors, cheating allowed
    # 2. All sectors, cheating banned
    # 3. Target sector, cheating allowed
    # 4. Target sector, cheating banned
    #
    # The metrics we want to collect involve quantifying how well
    # the forecast matched the target's historical performance 
    # following break out, as well as the goodness of the 
    # predictor stock charts probably defined by the LSEs. These
    # metrics can all be determined from the 6 outputs of
    # MatchFunction.match, pred - yeval allows for the quality
    # of the forecast to be measured; df_pats contains the LSEs
    # of the predictor stock charts which can be visualized in
    # varying ways (boxplot, lineplot)
    ##############################################################
    
    ##############################################################
    # use the chart pattern matching algorithm to output a forecast
    # 1. All sectors, cheating allowed
    ya1, pred1, yeval1, xvalst1, xvalsp1, df_pats1 = mf.match(df=df,pattern_length=pattern_length,
                                               pred_length=prediction_length,
                                               target=target,
                                               start_date=start_date,
                                               n_rand_start=20,
                                               random_seed=10,
                                               cheating=True)
    
    # Score the forecast
    #SE_forecast1 = mf.getLSE(pred1,yeval1)
    
    ##############################################################
    # 2. All sectors, cheating banned
    ya2, pred2, yeval2, xvalst2, xvalsp2, df_pats2 = mf.match(df=df,pattern_length=pattern_length,
                                               pred_length=prediction_length,
                                               target=target,
                                               start_date=start_date,
                                               n_rand_start=20,
                                               random_seed=10,
                                               cheating=False)
    
    # Score the forecast
    #SE_forecast2 = mf.getLSE(pred2,yeval2)
    
    ##############################################################
    # 3. Target sector, cheating allowed
    ya3, pred3, yeval3, xvalst3, xvalsp3, df_pats3 = mf.match(df=dfs,pattern_length=pattern_length,
                                               pred_length=prediction_length,
                                               target=target,
                                               start_date=start_date,
                                               n_rand_start=20,
                                               random_seed=10,
                                               cheating=True)
    
    #SE_forecast3 = mf.getLSE(pred3,yeval3)
    
    ##############################################################
    # 4. Target sector, cheating allowed
    ya4, pred4, yeval4, xvalst4, xvalsp4, df_pats4 = mf.match(df=dfs,pattern_length=pattern_length,
                                               pred_length=prediction_length,
                                               target=target,
                                               start_date=start_date,
                                               n_rand_start=20,
                                               random_seed=10,
                                               cheating=False)
    
    #SE_forecast4 = mf.getLSE(pred4,yeval4)
    
    df_pats1['Case'] = 'All Sectors, Cheat=T'
    df_pats2['Case'] = 'All Sectors, Cheat=F'
    df_pats3['Case'] = 'Target Sector, Cheat=T'
    df_pats4['Case'] = 'Target Sector, Cheat=F'
    
    all_pats = pd.concat([df_pats1,df_pats2,df_pats3,df_pats4])
    import seaborn as sns
    
    ##############################################################
    # Boxplot of LSE of predictor stocks by model variant
    plt.figure()
    bplot = sns.boxplot(y='LSE', x='Case', 
                     data=all_pats, 
                     width=0.5,
                     palette="colorblind")
    bplot.set_xticklabels(bplot.get_xticklabels(),rotation=30)
    bplot.set_ylim(0,5*(df_pats2['LSE'].mean()))
    bplot.set_title(target+" "+str(start_date)[:-9]+' to '+str(end_date)[:-9]
                    +"\n"+pattern_type)
    fig = bplot.get_figure()
    
    # Make sure there's a location to save off our figures
    image_dir = os.path.join(image_loc, target)
    if os.path.exists(image_dir):
        pass
    else:
        os.mkdir(image_dir)
    
    fig.savefig(os.path.join(image_dir,str(target)+'-LSE.png'),
                bbox_inches='tight', pad_inches=1)
    
    
    ##############################################################
    # Boxplot of scale factors of predictor stocks by model variant
    plt.figure()
    bplot1 = sns.boxplot(y='Sls', x='Case', 
                     data=all_pats, 
                     width=0.5,
                     palette="colorblind")
    bplot1.set_xticklabels(bplot.get_xticklabels(),rotation=30)
    bplot1.set_ylim(0,5)
    bplot1.set_title(target+" "+str(start_date)[:-9]+' to '+str(end_date)[:-9]
                     +"\n"+pattern_type)
    fig1 = bplot1.get_figure()
    fig1.savefig(os.path.join(image_dir,str(target)+'-SLS.png'),
                 bbox_inches='tight', pad_inches=1)
    
    ##############################################################
    # Plot the absolute values of the differences between the
    # 4 model variant forecasts and historical performance of target
    #mf.plotMatch(ya1, pred1, yeval1, xvalst1, xvalsp1,title=target,)
    
    D1 = pred1 - yeval1
    absD1 = []
    for i in range(len(D1)): 
        absD1.append(math.fabs(D1[i]))
        
    D2 = pred2 - yeval2
    absD2 = []
    for i in range(len(D2)): 
        absD2.append(math.fabs(D2[i]))
    
    D3 = pred3 - yeval3
    absD3 = []
    for i in range(len(D3)): 
        absD3.append(math.fabs(D3[i]))
        
    D4 = pred4 - yeval4
    absD4 = []
    for i in range(len(D4)): 
        absD4.append(math.fabs(D4[i]))
    
    plt.figure()
    xvals = np.arange(len(D1))    
    plt.plot(xvals, absD1, '-', label='All Sectors, Cheat=T')
    plt.plot(xvals, absD2, '-', label='All Sectors, Cheat=F')
    plt.plot(xvals, absD3, '-', label='Target Sector, Cheat=T')
    plt.plot(xvals, absD4, '-', label='Target Sector, Cheat=F')
    plt.legend(loc='best')
    plt.xlabel("Days")
    plt.ylabel("Absolute value of forecast - actual")
    plt.title(target+" "+str(start_date)[:-9]+' to '+str(end_date)[:-9]
              +"\n"+pattern_type)
    plt.savefig(os.path.join(image_dir,str(target)+'-DIF.png'),
                bbox_inches='tight', pad_inches=1)
    
    ##############################################################
    # Plot the 4 model variant forecasts and historical 
    # performance of target
    
    plt.figure()
    mf.plotMatch(ya1, pred1, yeval1, xvalst1, xvalsp1,
                 title=target+" "+str(start_date)[:-9]+' to '+str(end_date)[:-9]
              +"\n"+pattern_type,label='All Sectors, Cheat=T',xy=[-1,-1],
              color='orange')
    mf.plotMatch(ya2, pred2, yeval2, xvalst2, xvalsp2,
                 title=target+" "+str(start_date)[:-9]+' to '+str(end_date)[:-9]
              +"\n"+pattern_type,label='All Sectors, Cheat=F',xy=[-1,-1],
              color='pink')
    mf.plotMatch(ya3, pred3, yeval3, xvalst3, xvalsp3,
                 title=target+" "+str(start_date)[:-9]+' to '+str(end_date)[:-9]
              +"\n"+pattern_type, label='Target Sector, Cheat=T',xy=[-1,-1],
              color='y')
    mf.plotMatch(ya4, pred4, yeval4, xvalst4, xvalsp4,
                 title=target+" "+str(start_date)[:-9]+' to '+str(end_date)[:-9]
              +"\n"+pattern_type, label='Target Sector, Cheat=F',xy=[-1,-1],
              color='m',label_act=True)
    plt.legend(loc='best')
    plt.savefig(os.path.join(image_dir,str(target)+'-FOR.png'))