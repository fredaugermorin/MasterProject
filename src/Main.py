# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 23:58:10 2017

@author: FRED-PC
"""
from utils import Option

if __name__=="__main__":
    s= 36. #spot price
    sigma= 0.2 #volatility
    T= 1. #time to expiry
    k= 40. #strike price
    r= 0.06 #deterministic short term interest rate
    opt_type= 'PUT'
    n= 100000 #number of simulations
    m= int(T*50) #number of exercise points (default 50 per year in OG article)
    conf= 0.95 # confidence level for estimation
    
    # Test LSMC Algorithm
    
    opt= Option(opt_type, s, k, T, sigma, r)
    opt.valuation(n,m)
    opt.display(conf)


