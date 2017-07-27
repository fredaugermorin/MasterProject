# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 23:58:10 2017

@author: FRED-PC
"""
from utils import Option
import matplotlib.pyplot as plt
import numpy as np
import random
import csv

if __name__=="__main__":
    s= 36. #spot price
    sigma= 0.2 #volatility
    T= 1. #time to expiry
    k= 40. #strike price
    r= 0.06 #deterministic short term interest rate
    opt_type= 'PUT'
    n= 100 #number of simulations
    m= int(T*50) #number of exercise points (default 50 per year in OG article)
    conf= 0.95 # confidence level for estimation
    
    # Test LSMC Algorithm
    
    opt= Option(opt_type, s, k, T, sigma, r)
    opt.valuation(n,m)
    #opt.display(True)
    
    n = 100000
    m = 50
    timeSteps= np.arange(1,m+1,1)
    fieldNames= ['S','K','sig','r','T','m','n','price','std']
    numSample= 100000
    with open('results.csv', 'wb') as csvfile:
        resWriter = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        resWriter.writerow(fieldNames +  [str(t) for t in timeSteps] )
        
        for i in range(0,numSample):
            print(i)
            S = 30.+np.random.uniform(0,1)*20.
            K = 40.+np.random.uniform(0,1)*20.
            T = random.randint(1,8)*0.25
            r = np.random.uniform(0,1)*0.15
            sig = 0.05+np.random.uniform(0,1)*0.35

            
            opt= Option(opt_type,S,K,T,sig,r)
            try:
                price,se=opt.valuation(n,m)
            
                toWrite= [S,K,sig,r,T,m,n,price,se]
                for b in opt.exerciseBoudary:
                    toWrite.append(b)
            
                resWriter.writerow(toWrite)
            except:
                pass

