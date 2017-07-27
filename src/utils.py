# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 21:33:55 2017

@author: FRED-PC
"""
from __future__ import division

import abc
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import time

class MonteCarloSim(object):
    def __init__(self,n,m,seed):
        self.n = n
        self.m = m
        self.seed= seed
        
    def antitheticGBM(self,S,r,T,sigma):
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
        m= self.m
        n= self.n
        seed = self.seed
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
            
        def display(self,boundary=True):
            val,se= self.MCPrice
            z= norm.pdf(0.5+self.conf/2.)
            print("###   "+self.opt_type+" option value using LSMC   ###")
            print("Option Price: %.4f$  [%.4f, %.4f]\n")%(val,val-se*z/np.sqrt(self.MC.n),val+se*z/np.sqrt(self.MC.n))
            
            if boundary:
                plt.plot(np.arange(self.dt,self.T+self.dt,self.dt),self.exerciseBoudary)
                plt.ylabel('S')
                plt.xlabel('T')
                plt.title('Exercise Boundary for '+self.opt_type+' Option')
                plt.show()
                
            return 0
        
        def valuation(self,n,m,conf=0.95,seed=None):
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
            
            self.n= n
            self.m= m
            self.conf= conf
            
            self.exerciseBoudary= np.zeros(m)
            
            if seed is None:
                seed= int(time.time())
            
            self.MC = MonteCarloSim(self.n, self.m, seed)
            self.sims= self.MC.antitheticGBM(S,r,T,sig)
            sims = self.sims
            
            self.dt = T/m
            dt = self.dt
            #sims= antitheticGBM(S,r,T,sigma,n,m,seed)
            # cash flows matrix
            CFP= np.zeros((n,m+1))
            
            #set final cashflows as if option was european
            if self.opt_type=='PUT':
                CFP[:,-1]= np.vstack([K-sims[:,-1], np.zeros(n)]).max(axis=0)
                self.exerciseBoudary[-1]= sims[K>sims[:,-1],-1].max()
            elif self.opt_type== 'CALL':
                CFP[:,-1]= np.vstack([sims[:,-1]-K, np.zeros(n)]).max(axis=0)
                self.exerciseBoudary[-1]= sims[sims[:,-1]>K,-1].min()
            
            # Laguerre's polynomials
            L_0= lambda x: np.ones(x.shape)
            L_1= lambda x: 1.-x
            L_2= lambda x: 1./2.*(2.-4.*x-x**2.);
            
            #Proceed backwards in time
            for i in range(m-1,0,-1):
                
                if self.opt_type=='PUT':
                    exo= np.vstack([K-sims[:,i], np.zeros(n)]).max(axis=0)
                elif self.opt_type== 'CALL':
                    exo= np.vstack([sims[:,i]-K, np.zeros(n)]).max(axis=0)
                    
                atmIdx= np.where(exo>0.)[0] #indices of atm paths
                
                xP= (sims[atmIdx, i]/S) #value of underlying on atm paths
                yP= (CFP[atmIdx, i+1]*np.exp(-r*dt)) #pv of cashflow of atm paths
                
                coeffs = np.linalg.lstsq(np.vstack([L_0(xP), L_1(xP), L_2(xP)]).T,yP.T)[0]
                
                C= np.dot(np.vstack([L_0(xP), L_1(xP), L_2(xP)]).T, coeffs) #Exercise value obtained from regression
                
                exIdx= np.where(exo[atmIdx] >= C)[0] #indices of atm exercised paths
                try:
                    if self.opt_type=='PUT':
                        self.exerciseBoudary[i-1]= sims[atmIdx[exIdx],i].max()
                    elif self.opt_type== 'CALL':
                        self.exerciseBoudary[i-1]= sims[atmIdx[exIdx],i].min()
                except:
                        self.exerciseBoudary[i-1]= np.nan
                
                CFP[atmIdx[exIdx],i]= exo[atmIdx[exIdx]] #cashflows of exercised paths
                nIdx= np.setdiff1d(range(0,n), atmIdx[exIdx]) #indices of non exercised cashflows
                CFP[nIdx,i]= CFP[nIdx,i+1]*np.exp(-r*dt) #discount prior cashflow on the non exercised path
            
            CFP[:,0]= CFP[:,1]*np.exp(-r*dt) #compute PV of cashflows for all paths
            opt= np.mean(CFP[:,0]) #MC estimator of option price
            se= np.std(CFP[:,0]) #Variance of option prices on all paths
            
            self.MCPrice= [opt, se]
            
            return [opt, se]
            