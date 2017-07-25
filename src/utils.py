# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 21:33:55 2017

@author: FRED-PC
"""

import abc
import numpy as np

class MonteCarlo(object):
    def __init__(self, n, m, Instrument):
        self.n = n
        self.m = m
        self.Instrument = Instrument
        self.simulations = np.zeros((n,m+1))
        
    def antithetic(self):
        pass
        
        
class Instrument(object):
    @abc.abstractmethod
    def valuation(self):
        """ performs pricing and sensitivity calculations"""
        
class Option(Instrument):
        def __init__(self, opt_type, S, K, T, vol, r, d=None, exer_type=None):
            self.opt_type = opt_type
            self.S= S
            self.K= K
            self.T= T
            self.vol = vol
            self.r = r
            self.d = d or 0.
            exer_type= exer_type or 'european'
            
        def LSMC(n,m):
            pass
            