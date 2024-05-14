# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 21:02:44 2019

@author: Ali
"""
import numpy as np
import pandas as pd
import math as m
import FUNCTIONS as ft

k1 = 65553901281543303121104986366015842821
k2 = 340441033311546381796461911133388310263

FILE ="D:/power.csv"
Frequency = 3*10**6    # Hz
T = (1/Frequency) # seconds
SR = 1 * 10**6    # samples per Frame 
FR = 22 * 10**-3  # full frame 22 mSeconds
OFFSET = 100      # Start reading from here
Samples = SR/FR   # Samples per second 
Smpl_p= m.floor(T*Samples) #
PD  = (155*Smpl_p)    #
PA  = (370*Smpl_p)    # 
LEN = PD+PA+2*Smpl_p    # Length of a block of data
Len_d0 = PD+Smpl_p
Len_da = PD+PA+Smpl_p
Data = pd.read_csv(FILE)
LENGTH = Data.shape[0]
Power_Data = Data.Power

Lngt   = Data.shape[0]       # length of data read from file
L      = m.floor((LENGTH-OFFSET)/LEN)
DF     = Power_Data[OFFSET:L*LEN+OFFSET]

_len = len(ft._Dbls(k1,k2))
D_LEN = [0]*_len
for i in range(_len-1,-1,-1):
    if(ft._Dbls(k1,k2)[i] == 0):
        D_LEN[i] = Len_d0
    else:
        D_LEN[i] = Len_da
        

my_df = pd.DataFrame([0]*L*LEN)

for i in range(_len-1, -1 ,-1):
    P_data = np.array(DF[0:D_LEN[i]])
    Zero = np.array([0]*(Len_da-Len_d0))
    if (P_data.shape[0] < 7905):
          P_data=np.concatenate((P_data,Zero),axis=0)
    ft.add_row(my_df, P_data)
    
mydata = pd.read_csv("C:\\Users\\Ali\\Downloads\\Power_A2.csv", sep=',')

