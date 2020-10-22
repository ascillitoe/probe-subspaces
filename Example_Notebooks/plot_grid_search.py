#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  
from matplotlib import cm

mach_list = [0.3,0.4,0.6,0.7,0.8]
m_list    = [1,2,3,4] # reduced dimensions to try
k_list    = [2,3]   # poly order
nmach = len(mach_list)
nm    = len(m_list)
nk    = len(k_list)

y1_r2 = np.load('Yp_results.npy')
y2_r2 = np.load('Rr_results.npy')

# imach, im, ik
plt.figure()
plt.plot(m_list,y1_r2[0,:,0],'go-',label='$Y_p$, k = 2')
plt.plot(m_list,y1_r2[0,:,1],'go--',label='$Y_p$, k = 3',mfc='none',mew=2)
plt.plot(m_list,y2_r2[0,:,0],'bo-',label='$R_r$, k = 2')
plt.plot(m_list,y2_r2[0,:,1],'bo--',label='$R_r$, k = 3',mfc='none',mew=2)
plt.xlim([0.95,4.05])
plt.ylim([0.55,1.01])
plt.xticks([1,2,3,4])
plt.xlabel('Reduced dimension, $m$')
plt.ylabel('$R^2$')
plt.legend(loc='lower right')

plt.figure()
plt.plot(m_list,y1_r2[-1,:,0],'go-',label='$Y_p$, k = 2')
plt.plot(m_list,y1_r2[-1,:,1],'go--',label='$Y_p$, k = 3',mfc='none',mew=2)
plt.plot(m_list,y2_r2[-1,:,0],'bo-',label='$R_r$, k = 2')
plt.plot(m_list,y2_r2[-1,:,1],'bo--',label='$R_r$, k = 3',mfc='none',mew=2)
plt.xlim([0.95,4.05])
plt.ylim([0.55,1.01])
plt.xticks([1,2,3,4])
plt.xlabel('Reduced dimension, $m$')
plt.ylabel('$R^2$')
plt.legend(loc='lower right')

plt.show()
