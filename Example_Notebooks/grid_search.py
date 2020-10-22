#!/usr/bin/env python
# coding: utf-8

# In[147]:


#%matplotlib notebook
from equadratures import *
import numpy as np
import pandas as pd
from scipy.stats import linregress
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  
from matplotlib import cm
from sklearn.metrics import r2_score, mean_squared_error

Xorig = np.load('X.npy')
y1 = np.load('y1.npy')
y2 = np.load('y2.npy')

nrow, dim = np.shape(Xorig)

X = subspaces.standardise(Xorig)

m_list    = [1,2,3,4] # reduced dimensions to try
k_list    = [1,2,3]   # poly order
nm    = len(m_list)
nk    = len(k_list)

y1_r2   = np.zeros([nm,nk])
y2_r2   = np.zeros([nm,nk])  
y1_rmse = np.zeros([nm,nk]) 
y2_rmse = np.zeros([nm,nk]) 


s1 = Parameter(distribution='uniform', lower=-1., upper=1., order=2)
s2 = Parameter(distribution='uniform', lower=-1., upper=1., order=2)
myparameters1 = [s1 for _ in range(0, dim)]
myparameters2 = [s2 for _ in range(0, dim)]
#myparameters2 = [Parameter(distribution='custom',data=Xtrain[:,i], order=3) for i in range(0, dim)]

mybasis = Basis('total-order')

for im, m in enumerate(m_list):
    print('\t%i: m =  %.1f' %(im, m))

    for ik, k in enumerate(k_list):
        if(k==1 and m>1):
            y1_r2[im,ik] = 0.0
            y1_rmse[im,ik] = 0.0
            y2_r2[im,ik] = 0.0
            y2_rmse[im,ik] = 0.0
        else:
            print('\t\t%i: k =  %.1f' %(ik, k))

            y1temp = 0.0
            y2temp = 0.0
            for r in range(5):
                # get subspaces
                mysubspace1 = Subspaces(method='variable-projection',sample_points=X,sample_outputs=y1,polynomial_degree=k, subspace_dimension=m)
                mysubspace2 = Subspaces(method='variable-projection',sample_points=X,sample_outputs=y2,polynomial_degree=k, subspace_dimension=m)

                W1 = mysubspace1.get_subspace()
                W2 = mysubspace2.get_subspace()
                active_subspace1 = W1[:,0:m]
                active_subspace2 = W2[:,0:m]
                u1 = X @ active_subspace1
                u2 = X @ active_subspace2
                y1test = y1
                y2test = y2
                subpoly1 = mysubspace1.get_subspace_polynomial()
                subpoly2 = mysubspace2.get_subspace_polynomial()

                # Predict y1 and y2 at test points using subspace poly. Get r2 and RMSE
                y1pred = subpoly1.get_polyfit(u1)
                y2pred = subpoly2.get_polyfit(u2)

                y1temp = r2_score(y1test, y1pred)
                if(y1temp>=y1_r2[im,ik]):
                    y1_r2[im,ik] = y1temp
                    y1_rmse[im,ik] = np.sqrt(mean_squared_error(y1test, y1pred))

                y2temp = r2_score(y2test, y2pred)
                if(y2temp>=y2_r2[im,ik]):
                    y2_r2[im,ik] = y2temp
                    y2_rmse[im,ik] = np.sqrt(mean_squared_error(y2test, y2pred))

print('Yp')
data1 = pd.DataFrame(y1_r2[:,:],columns=k_list,index=m_list)
print(data1)
print('Rr')
data2 = pd.DataFrame(y2_r2[:,:],columns=k_list,index=m_list)
print(data2)

np.save('Yp_results.npy',y1_r2)
np.save('Rr_results.npy',y2_r2)

plt.show()
