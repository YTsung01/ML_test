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

data=pd.read_hdf("FallAllD.h5")  #可以看到資料
newdata=data.copy()
nd=(newdata[newdata["Device"].isin(["Waist"])]).reset_index() #nd為只有腰部的資料
  
k1=(nd['ActivityID']).values.reshape(-1, 1)
for i in range(0,1798):
    if k1[i]>=100:
        k1[i]=1
        
    else:
        k1[i]=0         #將大於100的變成1 小於100的變成0        
TargetVector=k1   

#%%
waist=[]
for i in range(len(nd)):
    test01=nd['Acc'][i]
    waist.append(test01)         #將3軸ACC的資料取出 存在waist裡面
    
     
norm=[]
for i in range(len(waist)):
    a0=np.array(waist[i][:,0], dtype='int64')**2
    a1=np.array(waist[i][:,1], dtype='int64')**2
    a2=np.array(waist[i][:,2], dtype='int64')**2
    test02=np.sqrt(a0+a1+a2)
    norm.append(test02)
    
hori=[]
for i in range(len(waist)):
    a1=np.array(waist[i][:,1], dtype='int64')**2
    a2=np.array(waist[i][:,2], dtype='int64')**2
    test02_1=np.sqrt(a1+a2)
    hori.append(test02_1) 
    
verti=[]
for i in range(len(waist)):
    a0=np.array(waist[i][:,0], dtype='int64')**2
    a1=np.array(waist[i][:,1], dtype='int64')**2
    test02_2=np.sqrt(a0+a1)
    verti.append(test02_2)
    
    
waist=np.array(waist)       
norm=np.array(norm)
hori=np.array(hori)
verti=np.array(verti)

d=np.reshape(norm,(1798,4760,1))
waist=np.concatenate((waist,d),axis=2)   #將waist3軸ACC的資料和norm合併成一個矩陣

d=np.reshape(hori,(1798,4760,1))
waist=np.concatenate((waist,d),axis=2)
d=np.reshape(verti,(1798,4760,1))
waist=np.concatenate((waist,d),axis=2)

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
        type01=np.mean(waist[i,:,j]) 
        featuremean.append(type01)
        type02=np.std(waist[i,:,j])
        featurestd.append(type02)
        type03=np.var(waist[i,:,j])
        featurevar.append(type03)
        type04=np.max(waist[i,:,j])
        featuremax.append(type04)
        type05=np.min(waist[i,:,j])
        featuremin.append(type05)
        type06=kurtosis(waist[i,:,j])
        featurekurtosis.append(type06)
        type07=skew(waist[i,:,j])
        featureskew.append(type07)
        type08=type04-type05
        featurerange.append(type08)
        
              
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
xycc=[]
xzcc=[]
yzcc=[]
nvcc=[]
nhcc=[]
vhcc=[]
for i in range(len(waist)):
    pccs01 = np.corrcoef(waist[i,:,0], waist[i,:,1]) #x軸跟y軸的Correlation coefficien
    pccs01=pccs01[0,1]
    xycc.append(pccs01)
    pccs02 = np.corrcoef(waist[i,:,0], waist[i,:,2]) #x軸跟z軸的Correlation coefficien
    pccs02=pccs02[0,1]
    xzcc.append(pccs02)
    pccs03 = np.corrcoef(waist[i,:,1], waist[i,:,2]) #y軸跟z軸的Correlation coefficien
    pccs03=pccs03[0,1]
    yzcc.append(pccs03)
    pccs04 = np.corrcoef(waist[i,:,3], waist[i,:,5]) #norm軸跟verti軸的Correlation coefficien
    pccs04=pccs04[0,1]
    nvcc.append(pccs04)
    pccs05 = np.corrcoef(waist[i,:,3], waist[i,:,4]) #norm軸跟hori軸的Correlation coefficien
    pccs05=pccs05[0,1]
    nhcc.append(pccs05)
    pccs06 = np.corrcoef(waist[i,:,5], waist[i,:,4]) #verti軸跟hori軸的Correlation coefficien
    pccs06=pccs06[0,1]
    vhcc.append(pccs06)

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
rr=np.rot90(feature,k=1,axes=(0,1))
rrr=np.flipud(TargetVector)
from sklearn import metrics
from sklearn.svm import SVC
X=rr
Y=rrr
model = SVC()
model.fit(X,Y)
expected = Y
predicted = model.predict(X)
print(metrics.classification_report(expected ,predicted))
print(metrics.confusion_matrix(expected ,predicted))

#%%
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

