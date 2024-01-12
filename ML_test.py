#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:11:28 2023

@author: songyuting
"""

import cmath
import numpy as np
import pandas as pd
from scipy.stats import kurtosis
from scipy.stats import skew

data=pd.read_hdf("FallAllD.h5")  
waist_data=(data[data["Device"].isin(["Waist"])]).reset_index() #waist_data為只有腰部的資料
  
TargetVector=(waist_data['ActivityID']).values.reshape(-1, 1)
for i in range(0,1798):
    if TargetVector[i]>=100:
        TargetVector[i]=1
        
    else:
        TargetVector[i]=0         #將大於100的變成1 小於100的變成0        


#%%
waist=[]
for i in range(len(waist_data)):
    acc=waist_data['Acc'][i]
    waist.append(acc)         #將3軸ACC的資料取出 存在waist裡面
    
     
norm=[]
for i in range(len(waist)):
    x=np.array(waist[i][:,0], dtype='int64')**2 #取xyz軸的平方
    y=np.array(waist[i][:,1], dtype='int64')**2
    z=np.array(waist[i][:,2], dtype='int64')**2
    square=np.sqrt(x+y+z)   #xyz軸的平方 開根號
    norm.append(square)
    
hori=[]
for i in range(len(waist)):
    y=np.array(waist[i][:,1], dtype='int64')**2
    z=np.array(waist[i][:,2], dtype='int64')**2
    square=np.sqrt(y+z)
    hori.append(square) 
    
verti=[]
for i in range(len(waist)):
    x=np.array(waist[i][:,0], dtype='int64')**2
    y=np.array(waist[i][:,1], dtype='int64')**2
    square=np.sqrt(x+y)
    verti.append(square)
    
    
waist=np.array(waist)       
norm=np.array(norm)
hori=np.array(hori)
verti=np.array(verti)

data_reshape=np.reshape(norm,(1798,4760,1))
waist=np.concatenate((waist,data_reshape),axis=2)   #將waist3軸ACC的資料和norm合併成一個矩陣

data_reshape=np.reshape(hori,(1798,4760,1))
waist=np.concatenate((waist,data_reshape),axis=2)
data_reshape=np.reshape(verti,(1798,4760,1))
waist=np.concatenate((waist,data_reshape),axis=2)

#%%  取feacture
featuremean=[]
featurestd=[]
featurevar=[]
featuremax=[]
featuremin=[]
featurekurtosis=[]
featureskew=[]
featurerange=[]

for j in range(6):
    for i in range(len(waist)):
        data_mean=np.mean(waist[i,:,j]) 
        featuremean.append(data_mean)
        data_std=np.std(waist[i,:,j])
        featurestd.append(data_std)
        data_var=np.var(waist[i,:,j])
        featurevar.append(data_var)
        data_max=np.max(waist[i,:,j])
        featuremax.append(data_max)
        data_min=np.min(waist[i,:,j])
        featuremin.append(data_min)
        data_kurtosis=kurtosis(waist[i,:,j])
        featurekurtosis.append(data_kurtosis)
        data_skew=skew(waist[i,:,j])
        featureskew.append(data_skew)
        data_range=data_max-data_min
        featurerange.append(data_range)
        
              
featuremean=np.array(featuremean)        
featuremean=np.reshape(featuremean,(6,1798))
featurestd=np.array(featurestd)        
featurestd=np.reshape(featurestd,(6,1798))  
featurevar=np.array(featurevar)        
featurevar=np.reshape(featurevar,(6,1798))  
featuremax=np.array(featuremax)        
featuremax=np.reshape(featuremax,(6,1798))     
featuremin=np.array(featuremin)        
featuremin=np.reshape(featuremin,(6,1798))  
featurekurtosis=np.array(featurekurtosis)        
featurekurtosis=np.reshape(featurekurtosis,(6,1798))    
featureskew=np.array(featureskew)        
featureskew=np.reshape(featureskew,(6,1798))
featurerange=np.array(featurerange)        
featurerange=np.reshape(featurerange,(6,1798))   

#%% 取feacture_Correlation coefficien
xycc=[]  #x軸跟y軸的Correlation coefficien
xzcc=[]  #x軸跟z軸的Correlation coefficien
yzcc=[]  #y軸跟z軸的Correlation coefficien
nvcc=[]  #norm軸跟verti軸的Correlation coefficien
nhcc=[]  #norm軸跟hori軸的Correlation coefficien
vhcc=[]  #verti軸跟hori軸的Correlation coefficien
for i in range(len(waist)):
    pccs = np.corrcoef(waist[i,:,0], waist[i,:,1]) #x軸跟y軸的Correlation coefficien
    pccs=pccs[0,1]
    xycc.append(pccs)
    pccs = np.corrcoef(waist[i,:,0], waist[i,:,2]) #x軸跟z軸的Correlation coefficien
    pccs=pccs[0,1]
    xzcc.append(pccs)
    pccs = np.corrcoef(waist[i,:,1], waist[i,:,2]) #y軸跟z軸的Correlation coefficien
    pccs=pccs[0,1]
    yzcc.append(pccs)
    pccs = np.corrcoef(waist[i,:,3], waist[i,:,5]) #norm軸跟verti軸的Correlation coefficien
    pccs=pccs[0,1]
    nvcc.append(pccs)
    pccs = np.corrcoef(waist[i,:,3], waist[i,:,4]) #norm軸跟hori軸的Correlation coefficien
    pccs=pccs[0,1]
    nhcc.append(pccs)
    pccs = np.corrcoef(waist[i,:,5], waist[i,:,4]) #verti軸跟hori軸的Correlation coefficien
    pccs=pccs[0,1]
    vhcc.append(pccs)

xycc=np.array(xycc) 
xycc=np.reshape(xycc,(1,1798))
xzcc=np.array(xzcc)     
xzcc=np.reshape(xzcc,(1,1798))
yzcc=np.array(yzcc)
yzcc=np.reshape(yzcc,(1,1798))
nvcc=np.array(nvcc)
nvcc=np.reshape(nvcc,(1,1798))
nhcc=np.array(nhcc)
nhcc=np.reshape(nhcc,(1,1798))
vhcc=np.array(vhcc)
vhcc=np.reshape(vhcc,(1,1798))

feature=[]
feature=np.concatenate((featuremean,featurestd,featurevar,featuremax,\
                        featuremin,featurekurtosis,featureskew,featurerange,\
                        xycc,xzcc,yzcc,nvcc,nhcc,vhcc),axis=0)
                             
    
#%%    
X=np.rot90(feature,k=1,axes=(0,1))
Y=np.flipud(TargetVector)
from sklearn import metrics
from sklearn.svm import SVC
model = SVC()
model.fit(X,Y)
expected = Y
predicted = model.predict(X)
print(metrics.classification_report(expected ,predicted))
print(metrics.confusion_matrix(expected ,predicted))

#%%LeaveOneOut
 
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(X)

output=[]
for train_index, test_index in loo.split(Y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index] 
   
    model = SVC()
    model.fit(X_train,Y_train.ravel())
    expected = Y_test
    predicted = model.predict(X_test)
    output.append(predicted)         #將預測結果存在一個新的矩陣
      
print(metrics.classification_report(Y,output))
print(metrics.confusion_matrix(Y,output))    
