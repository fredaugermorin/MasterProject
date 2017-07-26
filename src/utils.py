# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 21:33:55 2017

@author: FRED-PC
"""
from __future__ import division

import abc
import numpy as np
from scipy.stats import norm
import time

class MonteCarloSim(object):
    def __init__(self):
        pass

    def antitheticGBM(self,S,r,T,sigma,n,m,seed=None):
        """
        Method that computes antithetic paths for an asset
        following Geometric Brownian motion dynamics
        ## Input:
                S: Spot Price
                r: deterministic spot interest rate
                T: Option maturity in years
                sigma: underlyer's volatility
                n: number of total simulation trajectories
                m: number of time steps
        ## Output:
                sims: a n by m+1 matrix containing the simulated paths
        """
        self.m= m
        self.n= n
        if seed is None:
            seed= int(time.time())
    
        np.random.seed(seed)
        dt = T/m #Time Step
    
        sims = np.ones((n,m+1))*S # Initial value is spot price
        
        noise = np.random.normal(0,1,(int(n/2),m))
        R_plus = np.exp((r-sigma**2/2)*dt+sigma*np.sqrt(dt)*noise)
        R_moins = np.exp((r-sigma**2/2)*dt+sigma*np.sqrt(dt)*-noise)
        
        sims[:int(n/2),1:]= sims[:int(n/2),1:]*np.cumprod(R_plus,axis=1)
        sims[int(n/2):,1:]= sims[int(n/2):,1:]*np.cumprod(R_moins,axis=1)
                   
        self.sims=sims
        
        return self.sims
        
        
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
            exer_type= exer_type or 'american'
        
        def display(self,conf=0.95):
            val,se= self.MCPrice
            z= norm.pdf(0.5+conf/2.)
            print("### Pricing a "+self.opt_type+" option using LSMC ###")
            print("Option Price: %.4f$  [%.4f, %.4f]\n")%(val,val-se*z/np.sqrt(self.MC.n),val+se*z/np.sqrt(self.MC.n))
            return 0
        
        def valuation(self,n,m,seed=None):
            """
            Function that computes the value of an american option
            using the Longstaff Schwartz (2001) least squares Monte Carlo
            Algorithm
            ## Input:
                    S: Spot Price
                    K: Strike Price
                    r: deterministic spot interest rate
                    T: Option maturity in years
                    sigma: underlyer's volatility
                    n: number of simulation trajectories
                    m: number of time steps
                    opt_type: option type 'CALL' or 'PUT'
                    seed: random number generator seed for reproduction
            ## Output:
                    price: an array containing Callprice, PutPrice, StdDev Call, Std Dev Put
            """
            S= self.S
            K= self.K
            r= self.r
            sig= self.vol
            T= self.T
            if seed is None:
                seed= int(time.time())
            
            self.MC = MonteCarloSim()
            sims= self.MC.antitheticGBM(S,r,T,sig,n,m,seed=None)
            
            dt = T/m
            
            #sims= antitheticGBM(S,r,T,sigma,n,m,seed)
            # cash flows matrix
            CFP= np.zeros((n,m+1))
            
            if self.opt_type=='PUT':
                CFP[:,-1]= np.vstack([K-sims[:,-1], np.zeros(n)]).max(axis=0)
            elif self.opt_type== 'CALL':
                CFP[:,-1]= np.vstack([sims[:,-1]-K, np.zeros(n)]).max(axis=0)
            
            # Laguerre's polynomials
            L_0= lambda x: np.ones(x.shape)
            L_1= lambda x: 1.-x
            L_2= lambda x: 1./2.*(2.-4.*x-x**2.);
            
            for i in range(m-1,0,-1):
                if self.opt_type=='PUT':
                    exo= np.vstack([K-sims[:,i], np.zeros(n)]).max(axis=0)
                elif self.opt_type== 'CALL':
                    exo= np.vstack([sims[:,i]-K, np.zeros(n)]).max(axis=0)
                    
                atmIdx= np.where(exo>0.)[0]
                
                xP= (sims[atmIdx, i]/S)
                yP= (CFP[atmIdx, i+1]*np.exp(-r*dt))
                
                coeffs = np.linalg.lstsq(np.vstack([L_0(xP), L_1(xP), L_2(xP)]).T,yP.T)[0]
                
                C= np.dot(np.vstack([L_0(xP), L_1(xP), L_2(xP)]).T, coeffs)
                
                exIdx= np.where(exo[atmIdx] > C)[0]
                
                CFP[atmIdx[exIdx],i]= exo[atmIdx[exIdx]]
                nIdx= np.setdiff1d(range(0,n), atmIdx[exIdx])
                CFP[nIdx,i]= CFP[nIdx,i+1]*np.exp(-r*dt)
            
            CFP[:,0]= CFP[:,1]*np.exp(-r*dt)
            opt= np.mean(CFP[:,0])
            se= np.std(CFP[:,0])
            
            self.MCPrice= [opt, se]
            
            return [opt, se]
            