#! /home/linot/anaconda3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  6 13:44:25 2022

@author: Alec
"""

import os
import sys
import math

import numpy as np
import pickle
import matplotlib as pl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt

# import tensorflow as tf
# from tensorflow import keras
# import tensorflow.keras.backend as K

import scipy.io
from sklearn.utils.extmath import randomized_svd
from numpy import genfromtxt

import time

import torch
import torch.nn as nn
import torch.optim as optim

# from scipy.integrate import odeint
# from sklearn.cluster import KMeans

import os

if os.environ['PATH'][-6:]!='texbin':
    os.environ['PATH'] = os.environ['PATH'] + ':/Library/TeX/texbin'

import matplotlib.font_manager

#rc('font',**{'family':'serif','serif':['Computer Modern Roman'],'size':10})
# rc('text', usetex=True)

# This appears to get the correct font!!!!!
import matplotlib as mpl
import matplotlib.font_manager as font_manager
mpl.rcParams['font.family']='serif'
cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
mpl.rcParams['font.serif']=cmfont.get_name()
mpl.rcParams['mathtext.fontset']='cm'
mpl.rcParams['axes.unicode_minus']=False
mpl.rcParams['font.size']=12


###############################################################################
# Classes
###############################################################################
 
    
class observe(nn.Module):
    
    def __init__(self,trunc,N):
        super(observe, self).__init__()
        # NN Arch
        self.obs = nn.Sequential(
            nn.Linear(N, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, 100),
            nn.ELU(),
            nn.Linear(100, trunc),
            nn.ELU()
        )
        
    def forward(self, y):
        # Compute the dictionary
        dic=self.obs(y)
        return torch.cat((y,dic), dim=-1)

###############################################################################
# Functions
###############################################################################

    
###############################################################################
# Main Function
###############################################################################
if __name__=='__main__':

    # Perform the training test split

    a_data= genfromtxt('/home/costanteamor/KS/data/x_data_duffing_t_01_IC_1000_len_100.csv', delimiter=',')
    b_data = genfromtxt('/home/costanteamor/KS/data/y_data_duffing_t_01_IC_1000_len_100.csv', delimiter=',')


    frac=.8

    [M,N]=a_data.shape

    a_train=a_data[:round(M*frac),:]
    a_trainf=a_data[1:round(M*frac)+1,:]
    a_test=a_data[round(M*frac):M,:]


    a_train= a_data  # X(t,)
    a_trainf= b_data #  X(t+dt,)


    true_y=torch.tensor(a_train[:10000,:])
    true_yf=torch.tensor(a_trainf[:10000,:])


 ###########################################################################
    # Plot results 
    ###########################################################################
    auto=torch.load('model.pt')



    pred=auto.forward(true_y)
    predf=auto.forward(true_yf)
   


    a_data_torch=torch.tensor(a_data[:,:])


    IC=19000
    K=torch.linalg.pinv(pred)@predf
    y0=auto.forward(a_data_torch[IC,:]).detach().numpy()


    Ktemp=K.detach().numpy()
    [lam,V]=np.linalg.eig(Ktemp)


    ys=[y0]
    for i in range(100):
        y0=Ktemp.T@y0
        ys.append(y0)
    
    
    comp=a_data_torch[IC:IC+100,:].detach().numpy()
    ys=np.asarray(ys)

    

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3.5*2, 2.5), sharex=False, sharey=True,dpi=400)

    # ax=plt.subplot(projection='3d')
    ax1.plot(comp[:,0], '-',color='black',linewidth=1.5,label='True')
    ax1.plot(ys[:,0],'--',color='red',linewidth=1.5,label='Model')
    ax2.plot(comp[:,1], '-',color='black',linewidth=1.5,label='True')
    ax2.plot(ys[:,1],'--',color='red',linewidth=1.5,label='Model')
    ax1.set_xlabel(r'$\delta t$')
    ax1.set_ylabel(r'$x_1$')
    ax2.set_xlabel(r'$\delta t$')
    ax2.set_ylabel(r'$x_2$')
    plt.legend(loc='best',prop={'size':12})
    plt.tight_layout()
    plt.savefig('traj.png')  



#--Eigenvalues


    x = np.linspace( -1 , 1 , 150 )
    y = np.linspace( -1 , 1 , 150 )
    a, b = np.meshgrid( x , y )
    C = a ** 2 + b ** 2 - 1



    fig, (ax1) = plt.subplots(1, 1, figsize=(3.3, 3), sharex=False, sharey=True,dpi=400)
    ax1.scatter(lam.real,lam.imag,s=30,color='black')
    ax1.set_xlabel(r'$Re(\lambda)$')
    ax1.set_ylabel(r'$Im(\lambda)$')
    # ax1.set_title('Eigenvalues: EDMD-DL')
    ax1.contour( a , b , C , [0], colors='red',linestyles= 'dashed')

    ax1.set_xlim([-1.05, 1.05])
    ax1.set_ylim([-1.05, 1.05])
    plt.tight_layout()

    fig.savefig("duffing_eig.pdf")
    

    
