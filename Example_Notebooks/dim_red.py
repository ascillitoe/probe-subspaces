#!/usr/bin/env python
# coding: utf-8

#%matplotlib notebook
from equadratures import *
import numpy as np
import pandas as pd
#from scipy.stats import linregress
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d  
from matplotlib import cm
from sklearn.metrics import r2_score, mean_squared_error
from joblib import dump, load
from matplotlib import rcParams
from funcs import grid_within_hull, mean_std_X, standardise

rcParams.update({'font.size': 22})
#rcParams.update({'font.family': 'sans-serif'})
#rcParams.update({'font.sans-serif': 'Helvetica'})
rcParams.update({'axes.edgecolor':'#333F4B'})
#rcParams.update({'axes.linewidth':1.5})
rcParams.update({'xtick.color':'#333F4B'})
rcParams.update({'ytick.color':'#333F4B'})


#Xbaseline = np.array([0.5,-0.5,0.15,0,1,1,0]).T
#Xbaseline = np.array([0,0,0,0,1,1,1/3]).T
Xbaseline = np.array([0,0.428,-0.429,0,1,1,1/3]).T

#Xbaseline = np.array([0.35519312,-0.51331609,0.16345546,0.65717614,0.97631316,0.75267879,0.00668773]).T

Xorig = np.load('X.npy')
y1 = np.load('y1.npy')*100
y2 = np.load('y2.npy')
nrow, dim = np.shape(Xorig)
X, min_value, max_value = standardise(Xorig)
print(min_value,max_value)
Xheader = ['Hole $\oslash$','Kiel $\oslash_{inner}$','Kiel $\oslash_{outer}$','Kiel lip fwd/back','Hole angle','Hole fwd/back','Hole ellipse'] 

#Xbaseline,_,_ = standardise(Xbaseline,min_value=min_value,max_value=max_value)
#Xbaseline = Xbaseline.flatten()
#print(Xbaseline)

# get subspaces
subdim1 = 1
subdim2 = 2
k1 = 1
k2 = 3
#mysubspace1 = Subspaces(method='variable-projection',sample_points=X,sample_outputs=y1,polynomial_degree=k1, subspace_dimension=subdim1)
#mysubspace2 = Subspaces(method='variable-projection',sample_points=X,sample_outputs=y2,polynomial_degree=k2, subspace_dimension=subdim2)
#dump(mysubspace1,'Yp_subspace.joblib')
#dump(mysubspace2,'Rr_subspace.joblib')
mysubspace1 = load('Yp_subspace.joblib')
mysubspace2 = load('Rr_subspace.joblib')

W1 = mysubspace1.get_subspace()
W2 = mysubspace2.get_subspace()
active_subspace1 = W1[:,0:subdim1]
active_subspace2 = W2[:,0:subdim2]
u1 = X @ active_subspace1
u2 = X @ active_subspace2
y1test = y1
y2test = y2
subpoly1 = mysubspace1.get_subspace_polynomial()
subpoly2 = mysubspace2.get_subspace_polynomial()

# get baseline in subspaces
u1baseline = Xbaseline @ active_subspace1
u2baseline = Xbaseline @ active_subspace2
y1baseline = subpoly1.get_polyfit(u1baseline)
y2baseline = subpoly2.get_polyfit(u2baseline)

# Predict y1 and y2 at test points using subspace poly. Get r2 and RMSE
y1pred = subpoly1.get_polyfit(u1)
y2pred = subpoly2.get_polyfit(u2)
y1_r2 = r2_score(y1test, y1pred)
y2_r2 = r2_score(y2test, y2pred)
y1_rmse = np.sqrt(mean_squared_error(y1test, y1pred))
y2_rmse = np.sqrt(mean_squared_error(y2test, y2pred))

# pred vs true scatter plots
plt.figure()
plt.plot(y1test,y1pred,'C3o',ms=10,mec='k',mew=2)
intv = np.max(y1test)-np.min(y1test)
plt.xlim([np.min(y1test)-0.05*intv,np.max(y1test)+0.05*intv])
plt.ylim([np.min(y1test)-0.05*intv,np.max(y1test)+0.05*intv])
plt.plot([np.min(y1test),np.max(y1test)],[np.min(y1test),np.max(y1test)],'k-',lw=3)
plt.annotate('$R^2 = %.3f$' %(y1_r2), (np.min(y1test)+0.1*intv,np.min(y1test)+0.8*intv))
plt.xlabel('True $O_{Y_p}$ $(\%)$')
plt.ylabel('Approximated $O_{Y_p}$ $(\%)$')

plt.figure()
plt.plot(y2test,y2pred,'C3o',ms=10,mec='k',mew=2)
intv = np.max(y2test)-np.min(y2test)
plt.xlim([np.min(y2test)-0.05*intv,np.max(y2test)+0.05*intv])
plt.ylim([np.min(y2test)-0.05*intv,np.max(y2test)+0.05*intv])
plt.plot([np.min(y2test),np.max(y2test)],[np.min(y2test),np.max(y2test)],'k',lw=3)
plt.annotate('$R^2 = %.3f$' %(y2_r2), (np.min(y2test)+0.1*intv,np.min(y2test)+0.8*intv))
plt.xlabel('True $O_{R_r}$')
plt.ylabel('Approximated $O_{R_r}$')

# 2d and 3d plots of poly (in subspace) and test points
N = 20
# Plot 1
rcParams.update({'font.size': 20})
fig2 = plt.figure(figsize=(6,6))
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('$u_Y$')
#ax2.set_xlabel('$\mathbf{u}=\mathbf{U}^T\mathbf{x}$')
ax2.set_ylabel('$O_{Y_p}$')
s1_samples = np.linspace(np.min(u1[:,0]), np.max(u1[:,0]), N)
PolyDiscreet = subpoly1.get_polyfit( s1_samples )
plt.plot(s1_samples,PolyDiscreet,c='C2',lw=3,label='Polynomial approx.')
ax2.scatter(u1, y1test, s=70, c=y1test, marker='o',edgecolors='k',linewidths=2,cmap=cm.coolwarm,label='Training designs')
intv = np.max(y1test)-np.min(y1test)
plt.ylim([np.min(y1test)-0.05*intv,np.max(y1test)+0.05*intv])
plt.legend()
#surf = ax2.plot_surface(S1, S2, PolyDiscreet, rstride=1, cstride=1, cmap=cm.gist_earth, linewidth=0, alpha=0.6, label='Tensor grid')
#ax2.plot(u1baseline,y1baseline ,'oC1',ms=20, mfc='none',mew=4)
#for i in range(u1.shape[0]):
#    ax2.annotate(str(i),[u1[i,0],y1test[i]])

# Plot 2
rcParams.update({'font.size': 20})
figRr = plt.figure(figsize=(10,10))
axRr = figRr.add_subplot(111, projection='3d')
axRr.scatter(u2[:,0], u2[:,1], y2test,  s=70, c=y2test, marker='o',edgecolors='k',linewidths=2)
axRr.set_xlabel('$u_{R,1}$')
axRr.set_ylabel('$u_{R,2}$')
axRr.set_xticks([-1,0,1])
axRr.set_yticks([-1,0,1])
axRr.set_zticks([0.002,0.004,0.006,0.008])
axRr.xaxis.labelpad=20
axRr.yaxis.labelpad=20
axRr.zaxis.labelpad=40
axRr.zaxis.set_rotate_label(False)
axRr.set_zlabel('$O_{R_r}$',rotation=0)
s1_samples = np.linspace(np.min(u2[:,0]), np.max(u2[:,0]), N)
s2_samples = np.linspace(np.min(u2[:,1]), np.max(u2[:,1]), N)
[S1, S2] = np.meshgrid(s1_samples, s2_samples)
S1_vec = np.reshape(S1, (N*N, 1))
S2_vec = np.reshape(S2, (N*N, 1))
samples = np.hstack([S1_vec, S2_vec])
PolyDiscreet = subpoly2.get_polyfit( samples )
PolyDiscreet = np.reshape(PolyDiscreet, (N, N))
surf = axRr.plot_surface(S1, S2, PolyDiscreet, rstride=1, cstride=1, cmap=cm.gist_earth, linewidth=0, alpha=0.5)
#intv = np.max(y2test)-np.min(y2test)
#ax3.set_zlim([np.min(y2test)-0.05*intv,np.max(y2test)+0.05*intv])

#from matplotlib import animation
#def rotate(angle):
#    axRr.view_init(azim=angle,elev=12)
#rot_animation = animation.FuncAnimation(figRr, rotate, frames=np.arange(0,362,2),interval=100)
#rot_animation.save('./rotation.gif', dpi=35, writer='imagemagick')
#plt.show()
#quit()

###########################################
# Plot weights of interesting u1,u2 vectors
###########################################
rcParams.update({'font.size': 18})

fig, ax = plt.subplots(figsize=(8,4))
k = W1[:,0]
y = np.arange(1,len(Xheader)+1)
cmap = cm.coolwarm
colors = [ cmap(x) for x in (k+1)/2 ]
ax.hlines(y=y,xmin=0,xmax=k, color=colors, alpha=0.5, linewidth=12)
ax.scatter(k,y,s=14**2,marker='o',c=k,cmap=cmap, vmin=-1,vmax=1,alpha=0.9)
print(k)
ax.set_xlabel('Components of $\mathbf{w}$', fontweight='black', color = '#333F4B')
ax.set_ylabel('')
ax.tick_params(axis='y', which='both',length=0)
plt.yticks(y, Xheader)
ax.set_yticks(np.append(y-0.5,np.max(y)+0.5), minor=True)
ax.grid(axis='y',color='#333F4B', alpha=0.4,linestyle='-',which='minor',linewidth=1.5)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
#ax.spines['bottom'].set_smart_bounds(True)
ax.spines['bottom'].set_position(('axes', -0.04)) 
ax.spines['left'].set_position(('axes', 0.015))
ax.vlines(0,0.5,np.max(y)+0.5,color='#333F4B', alpha=0.4,linestyle='--',linewidth=1.5)
ax.set_xlim([-0.1,1])
#x = np.arange(len(Xheader))  # the label locations
#width = 0.9 # the width of bars
#rects1 = ax.bar(x,k,width,edgecolor='k')
#ax.set_ylabel('Weights')
#ax.set_xticks(x)
#ax.set_xticklabels(Xheader,rotation=45,ha='right')
#ax.tick_params(axis='x',length=10, width=2)
#
#width = 0.45 # the width of bars
#k1 = W2[:,0]
#k2 = W2[:,1]
#fig = plt.figure(figsize=(6,6))
#ax = fig.add_subplot(111)
#rects1 = ax.bar(x-width/2,k1,width,label='$\mathbf{u}_{R_r}^a$',color='C2',edgecolor='k')
#rects2 = ax.bar(x+width/2,k2,width,label='$\mathbf{u}_{R_r}^b$',color='C7',edgecolor='k')
#ax.set_ylabel('Weights')
#ax.set_xticks(x)
#ax.set_xticklabels(Xheader,rotation=45,ha='right')
#ax.tick_params(axis='x',length=10, width=2)
#ax.legend()


##############################
# Zonotope plot 1, Rr subspace
##############################
rcParams.update({'font.size': 22})
fig3 = plt.figure(figsize=(8,6))
ax3 = fig3.add_subplot(111)
ax3.set_xlabel('$u_{R,1}$')
ax3.set_ylabel('$u_{R,2}$')
ax3.set_xlim([-1.5,1.5])
ax3.set_ylim([-1.5,1.5])
#ax3.set_xticks([-1.5,-1,-0.5,0,0.5,1,1.5])
#ax3.set_yticks([-1.5,-1,-0.5,0,0.5,1,1.5])

ax3.scatter(u2[:,0], u2[:,1], s=100, c=y2, marker='o',edgecolors='k',linewidths=2,vmin=0.001,vmax=0.009)#,vmin=0.001,vmax=0.008)
#for i in range(u2.shape[0]):
#    ax3.annotate(str(i),[u2[i,0],u2[i,1]])

U2 = mysubspace2.get_zonotope_vertices()
from funcs import polar_sort
U2 = polar_sort(U2)
#plt.plot(np.append(U2[:,0],U2[0,0]),np.append(U2[:,1],U2[0,1]),'k-',lw=3)
#levels = np.linspace(0.001,0.008,16)
#cont = ax3.tricontourf(u2[:,0], u2[:,1], y2,alpha=0.5,levels=levels)
cont = ax3.contourf(S1, S2, PolyDiscreet, rstride=1, cstride=1, cmap=cm.gist_earth, linewidth=1, levels=14,alpha=0.5,vmin=0.001,vmax=0.009)
ax3.contour(S1, S2, PolyDiscreet, colors='k',levels=14,alpha=0.5,vmin=0.001,vmax=0.009)

cbar = fig3.colorbar(cont,orientation="vertical",ticks=np.linspace(0.002,0.008,4))
cbar.ax.set_xlabel('$O_{R_r}$')

# baseline design
ax3.plot(u2baseline[0],u2baseline[1],'oC3',ms=15,mfc='none',mew=3)

# Plot vectors to study
v1s = u2[58,:]
v1e =  u2[112,:]
v1 = v1e - v1s
ax3.arrow(v1s[0], v1s[1], v1[0], v1[1],fc='C0',ec='r',linewidth=4,length_includes_head=True,head_length=0.1,head_width=0.1,alpha=1)
v1 /= np.sqrt(np.sum(v1*v1))
v2s = u2[115,:]
v2e = u2[26,:]
v2 = v2e - v2s
ax3.arrow(v2s[0], v2s[1], v2[0], v2[1],fc='C3',ec='w',linewidth=4,length_includes_head=True,head_length=0.1,head_width=0.1,alpha=1)
v2 /= np.sqrt(np.sum(v2*v2))
# Plot v1 and v2
print(v1)
print(v2)

# highlight designs A,B,C
ax3.plot(u2[58,0],u2[58,1],'sw',ms=17,mfc='none',mew=3)
ax3.annotate('$\mathbf{A}$', [u2[58,0]+0.05,u2[58,1]-0.08], fontsize=26,color='W',ha='left',va='top')
ax3.plot(u2[115,0],u2[115,1],'sw',ms=17,mfc='none',mew=3)
ax3.annotate('$\mathbf{B}$', [u2[115,0]+0.1,u2[115,1]-0.05], fontsize=26,color='W',ha='left',va='top')
ax3.plot(u2[112,0],u2[112,1],'sw',ms=17,mfc='none',mew=3)
ax3.annotate('$\mathbf{C}$', [u2[112,0]-0.05,u2[112,1]+0.08], fontsize=26,color='W',ha='center',va='bottom')
ax3.plot(u2[26,0],u2[26,1],'sw',ms=17,mfc='none',mew=3)
ax3.annotate('$\mathbf{D}$', [u2[26,0]+0.12,u2[26,1]], fontsize=26,color='W',ha='left',va='center')


#X = mysubspace2.get_samples_constraining_active_coordinates(10000,u2[57,:])
#n,dim = X.shape
#err = 99.0
#for d in range(n):
#    Xtemp = X[d,:]
#    u = Xtemp @ active_subspace1
#    Yp = subpoly1.get_polyfit( u )
#    ubaseline = Xbaseline @ active_subspace1
#    Ypbl = subpoly1.get_polyfit(ubaseline)
#    if np.abs(Yp-Ypbl) < err and Xtemp[4] > 0.9 and Xtemp[5] > 0.1 and np.abs(Xtemp[-1]) < 0.1:
#        err = np.abs(Yp-Ypbl)
#        dbest = d
#print(X[dbest,:])
#print(Yp,Ypbl)

u1_samples = np.linspace(v1s[0],v1e[0])
u2_samples = np.linspace(v1s[1],v1e[1])
v1_samples = np.column_stack((u1_samples,u2_samples))
#plt.plot(v1_samples[:,0],v1_samples[:,1],'--C0',lw=3)
plt.annotate('$\mathbf{v}_a$', [-0.45,0.64],fontsize=28,color='w')#,ha='left',va='bottom')

u1_samples = np.linspace(v2s[0],v2e[0])
u2_samples = np.linspace(v2s[1],v2e[1])
v2_samples = np.column_stack((u1_samples,u2_samples))
#plt.plot(v2_samples[:,0],v2_samples[:,1],'--C3',lw=3)
plt.annotate('$\mathbf{v}_b$',[0.36,-0.24],fontsize=28,color='w',ha='left',va='center')

# vector v1
k = v1[0]*W2[:,0]+v1[1]*W2[:,1]
k[-3] *= -1 # Reverse hole angle as updated definition for paper
fig, ax = plt.subplots(figsize=(8,4))
y = np.arange(1,len(Xheader)+1)
cmap = cm.coolwarm
colors = [ cmap(x) for x in (k+1)/2 ]
ax.hlines(y=y,xmin=0,xmax=k, color=colors, alpha=0.5, linewidth=12)
ax.scatter(k,y,s=14**2,marker='o',c=k,cmap=cmap, vmin=-1,vmax=1,alpha=0.9)

ax.set_xlabel('Components of $\mathbf{v}_a\mathbf{U}_R$', fontweight='black', color = '#333F4B')
ax.set_ylabel('')
ax.tick_params(axis='y', which='both',length=0)
plt.yticks(y, Xheader)
ax.set_yticks(np.append(y-0.5,np.max(y)+0.5), minor=True)
ax.grid(axis='y',color='#333F4B', alpha=0.4,linestyle='-',which='minor',linewidth=1.5)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
#ax.spines['bottom'].set_smart_bounds(True)
ax.spines['bottom'].set_position(('axes', -0.04)) 
ax.spines['left'].set_position(('axes', 0.015))
ax.vlines(0,0.5,np.max(y)+0.5,color='#333F4B', alpha=0.4,linestyle='--',linewidth=1.5)
ax.set_xlim([-1,1])

# vector v2
k = v2[0]*W2[:,0]+v2[1]*W2[:,1]
k[-3] *= -1 # Reverse hole angle as updated definition for paper
fig, ax = plt.subplots(figsize=(8,4))
y = np.arange(1,len(Xheader)+1)
cmap = cm.coolwarm
colors = [ cmap(x) for x in (k+1)/2 ]
ax.hlines(y=y,xmin=0,xmax=k, color=colors, alpha=0.5, linewidth=12)
ax.scatter(k,y,s=14**2,marker='o',c=k,cmap=cmap, vmin=-1,vmax=1,alpha=0.9)

ax.set_xlabel('Components of $\mathbf{v}_b\mathbf{U}_R$', fontweight='black', color = '#333F4B')
ax.set_ylabel('')
ax.tick_params(axis='y', which='both',length=0)
plt.yticks(y, Xheader)
ax.set_yticks(np.append(y-0.5,np.max(y)+0.5), minor=True)
ax.grid(axis='y',color='#333F4B', alpha=0.4,linestyle='-',which='minor',linewidth=1.5)
ax.spines['top'].set_color('none')
ax.spines['right'].set_color('none')
ax.spines['left'].set_smart_bounds(True)
#ax.spines['bottom'].set_smart_bounds(True)
ax.spines['bottom'].set_position(('axes', -0.04)) 
ax.spines['left'].set_position(('axes', 0.015))
ax.vlines(0,0.5,np.max(y)+0.5,color='#333F4B', alpha=0.4,linestyle='--',linewidth=1.5)
ax.set_xlim([-1,1])


####################################################
# Zonotope plot 2: with convex hull and grid plotted
####################################################
rcParams.update({'font.size': 22})
fig3 = plt.figure(figsize=(6,6))
ax3 = fig3.add_subplot(111)
ax3.set_xlabel('$u_{R,1}$')
ax3.set_ylabel('$u_{R,2}$')
ax3.set_xlim([-1.5,1.5])
ax3.set_ylim([-1.5,1.5])
#ax3.set_xticks([-2,-1,0,1,2])
#ax3.set_yticks([-2,-1,0,1,2])
#ax3.plot(np.append(U2[:,0],U2[0,0]),np.append(U2[:,1],U2[0,1]),'k-',lw=3)
new_samples = grid_within_hull(u2,ax=ax3,N=25)
new_samples = np.array(new_samples)

############################################
# Zonotope plot 3: with Yp projected onto it, and final designs shown
############################################
rcParams.update({'font.size': 20})

fig3 = plt.figure(figsize=(6,6))
ax3 = fig3.add_subplot(111)
ax3.set_xlabel('$u_{R,1}$')
ax3.set_ylabel('$u_{R,2}$')
ax3.set_xlim([-2,2])
ax3.set_ylim([-2,2])
ax3.set_xticks([-2,-1,0,1,2])
ax3.set_yticks([-2,-1,0,1,2])
ax3.plot(np.append(U2[:,0],U2[0,0]),np.append(U2[:,1],U2[0,1]),'k-',lw=3)

# For each u1,u2 coord in new_samples, generate ND designs
ND = 100#00
nrows_new = len(new_samples)
Xnew = np.empty([nrows_new,ND,dim])
Ypnew = np.empty(nrows_new)
Ypstd = np.empty(nrows_new)
Ypmean = np.empty(nrows_new)
for i in range(nrows_new):
    coord = new_samples[i,:]
    Xnew[i,:,:] = mysubspace2.get_samples_constraining_active_coordinates(ND,coord)

    # For each design, transform to Yp subspace and get predicted Yp from response surface
    Yptemp = np.zeros(ND)
    for d in range(ND):
        u1new = Xnew[i,d,:] @ active_subspace1
        Yptemp[d] = subpoly1.get_polyfit(u1new)

    # Take mean (or min?) Yp of 5 designs
#    Ypnew[i] = np.mean(Yptemp)
    Ypnew[i] = np.min(Yptemp)
    Ypstd[i] = np.std(Yptemp)
    Ypmean[i] = np.mean(Yptemp)

# Now do a contour plot of the new Yp on the Rr subspace
levels = np.linspace(0.07,0.14,8)
cont = ax3.tricontourf(new_samples[:,0], new_samples[:,1], Ypnew,alpha=0.8,levels=levels,cmap=cm.coolwarm)
ax3.tricontour(new_samples[:,0], new_samples[:,1], Ypnew,alpha=0.5,levels=levels,colors='k')
scat = ax3.scatter(u2[:,0], u2[:,1], s=100, c=y2, marker='o',edgecolors='k',linewidths=2)

#cbar1 = fig3.colorbar(cont,orientation="vertical",ticks=[0.08,0.1,0.12,0.14])
##cbar1.ax.set_title('$\min\left(O_{Y_p}\\right)$',fontsize=20,pad=20)
#cbar1.ax.set_title('$O_{Y_p}$',fontsize=20,pad=20)
#cax = fig3.add_axes([0.15, .95, 0.54, 0.03])
#cbar2 = fig3.colorbar(scat,orientation="horizontal",cax=cax)#,ticks=[0.08,0.1,0.12,0.14])
#cbar2.ax.set_ylabel('$O_{R_r}$',fontsize=20)#,pad=20)

# List of final u1-u2 coords in Rr subspace
#u2final = np.array([ 
#        [0.409,-1.411], #1
#        [0.598,-1.775], #2
#        [-0.322,-1.352],#3
#        [-1.037,-1.203],#4
#        [0.0, 1.642],   #5
#        [0.047,-1.03],  #6
#        [0.8,0.987],    #7
#        [-0.33,-0.33],  #8
#        [0.5,-0.33],    #9
#        ])
u2final = np.array([ 
        [-0.33,-0.33],  #8
        [0.047,-1.03],  #6
        [0.409,-1.411], #1
        [-0.322,-1.352],#3
        [-1.037,-1.203],#4
        [0.8,0.987],    #7
        ])



#ax3.plot(u2final[:,0],u2final[:,1],'xC3',ms=10,mew=2)
ax3.plot(u2baseline[0],u2baseline[1],'oC3',ms=15,mfc='none',mew=3)

#print('')
#for d in u2final:
#    print(d,subpoly2.get_polyfit(d))
#quit()

plt.show()
quit()


############################################
# Zonotope plot 4: variance of Yp
############################################
rcParams.update({'font.size': 20})

fig4 = plt.figure(figsize=(6,6))
ax4 = fig4.add_subplot(111)
ax4.set_xlabel('$u_{R,1}$')
ax4.set_ylabel('$u_{R,2}$')
ax4.set_xlim([-2,2])
ax4.set_ylim([-2,2])
ax4.set_xticks([-2,-1,0,1,2])
ax4.set_yticks([-2,-1,0,1,2])
ax4.plot(np.append(U2[:,0],U2[0,0]),np.append(U2[:,1],U2[0,1]),'k-',lw=3)

#levels = np.linspace(0.005,0.021,17)
levels=np.linspace(0,0.2,21)
cont = ax4.tricontourf(new_samples[:,0], new_samples[:,1], Ypstd/Ypmean,alpha=0.8,levels=levels,cmap=cm.coolwarm)
ax4.tricontour(new_samples[:,0], new_samples[:,1], Ypstd/Ypmean,alpha=0.5,levels=levels,colors='k')
cbar = fig4.colorbar(cont,orientation="vertical",ticks=[0.0,0.05,0.1,0.15,0.2])
cbar.ax.set_title('$\sigma\left(O_{Y_p}\\right)/\mathbb{E}\left(O_{Y_p}\\right)$',fontsize=20,pad=20)


##################
# Gen new designs
##################
ha = ['right','right','center','center','center','center']
va = ['center','top','top','top','top','bottom']
ho = [-5,-5,0,0,0,0]
vo = [0,-5,-5,-5,-5,5]
ND = 5
Xtosave = np.empty([u2final.shape[0]*ND,7+2])
row = 0
for c in range(u2final.shape[0]):
    ax3.annotate(str(c+1), (u2final[c,0],u2final[c,1]),fontsize=20,color='C3',ha=ha[c],va=va[c],xytext=(ho[c], vo[c]), textcoords='offset points')  # Label the design points on the zonotope plot

    print('Point %i, u1 = %.3f, u2 = %.3f' %(d,u2final[c,0], u2final[c,0]))
    Xfinal = mysubspace2.get_samples_constraining_active_coordinates(ND,u2final[c,:])
    Xfinal = subspaces.unstandardise(Xfinal,Xorig)
    for d in range(ND):
        with np.printoptions(precision=6,suppress=True):
            print('design %i:' %(d), Xfinal[d,:])
        Xtosave[row,0] = row
        Xtosave[row,1] = c+1
        Xtosave[row,2:] = Xfinal[d,:]
        row += 1
np.savetxt('newdesigns.csv', Xtosave, header='design #, u1-u2 point, ' + ", ".join(Xheader),fmt='%i, %i, %.6f, %.6f, %.6f, %.6f, %.6e, %.6e, %.6f') 


# Print mean and std of designs at certain points
Xheader = ['Hole diam','Kiel diam inner','Kiel diam outer','Kiel lip fwd/back','Hole angle','Hole fwd/back','Hole ellipse'] 
print('Design 40')
mean, std = mean_std_X(mysubspace2,u2[40,:])
for d in range(dim):
    print('%22s: %.5f +- %.5f' %(Xheader[d],mean[d], std[d]))

print('Design 2')
mean, std = mean_std_X(mysubspace2,u2[2,:])
for d in range(dim):
    print('%22s: %.5f +- %.5f' %(Xheader[d],mean[d], std[d]))

print('Design 86')
mean, std = mean_std_X(mysubspace2,u2[86,:])
for d in range(dim):
    print('%22s: %.5f +- %.5f' %(Xheader[d],mean[d], std[d]))

print('Point 1')
mean, std = mean_std_X(mysubspace2,u2final[0,:])
for d in range(dim):
    print('%22s: %.5f +- %.5f' %(Xheader[d],mean[d], std[d]))

print('Point 3')
mean, std = mean_std_X(mysubspace2,u2final[2,:])
for d in range(dim):
    print('%22s: %.5f +- %.5f' %(Xheader[d],mean[d], std[d]))

print('Point 5')
mean, std = mean_std_X(mysubspace2,u2final[4,:])
for d in range(dim):
    print('%22s: %.5f +- %.5f' %(Xheader[d],mean[d], std[d]))

print('00 point')
mean, std = mean_std_X(mysubspace2,[0,0])
for d in range(dim):
    print('%22s: %.5f +- %.5f' %(Xheader[d],mean[d], std[d]))

## Add 1-3 and 3-5 vectors
## 1-3
#v3s = u2final[0,:]
#v3e = u2final[2,:]
#v3 = v3e - v3s
#v3 /= np.sqrt(np.sum(v3*v3))
#print(v3)
#v4s = u2final[0,:]
#v4e = u2final[4,:]
#v4 = v4e - v4s
#v4 /= np.sqrt(np.sum(v4*v4))
#print(v4)
#u1_samples = np.linspace(v3s[0],v3e[0])
#u2_samples = np.linspace(v3s[1],v3e[1])
#v2_samples = np.column_stack((u1_samples,u2_samples))
#ax3.plot(v2_samples[:,0],v2_samples[:,1],'--C3',lw=3)
#ax3.annotate('$\mathbf{v}_a$',[v3e[0]+0.05,v3e[1]],fontsize=24,color='C3',ha='left',va='center')
#u1_samples = np.linspace(v4s[0],v4e[0])
#u2_samples = np.linspace(v4s[1],v4e[1])
#v2_samples = np.column_stack((u1_samples,u2_samples))
#ax3.plot(v2_samples[:,0],v2_samples[:,1],'--C3',lw=3)
#ax3.annotate('$\mathbf{v}_a$',[v4e[0]+0.05,v4e[1]],fontsize=24,color='C3',ha='left',va='center')
#
#
#k = v3[0]*W2[:,0]+v3[1]*W2[:,1]
#fig, ax = plt.subplots(figsize=(8,4))
#y = np.arange(1,len(Xheader)+1)
#cmap = cm.coolwarm
#colors = [ cmap(x) for x in (k+1)/2 ]
#ax.hlines(y=y,xmin=0,xmax=k, color=colors, alpha=0.5, linewidth=12)
#ax.scatter(k,y,s=14**2,marker='o',c=k,cmap=cmap, vmin=-1,vmax=1,alpha=0.9)
#ax.set_xlabel('Components of $\mathbf{v}_b\mathbf{w}$', fontweight='black', color = '#333F4B')
#ax.set_ylabel('')
#ax.tick_params(axis='y', which='both',length=0)
#plt.yticks(y, Xheader)
#ax.set_yticks(np.append(y-0.5,np.max(y)+0.5), minor=True)
#ax.grid(axis='y',color='#333F4B', alpha=0.4,linestyle='-',which='minor',linewidth=1.5)
#ax.spines['top'].set_color('none')
#ax.spines['right'].set_color('none')
#ax.spines['left'].set_smart_bounds(True)
##ax.spines['bottom'].set_smart_bounds(True)
#ax.spines['bottom'].set_position(('axes', -0.04)) 
#ax.spines['left'].set_position(('axes', 0.015))
#ax.vlines(0,0.5,np.max(y)+0.5,color='#333F4B', alpha=0.4,linestyle='--',linewidth=1.5)
#ax.set_xlim([-1,1])
#k = v4[0]*W2[:,0]+v4[1]*W2[:,1]
#fig, ax = plt.subplots(figsize=(8,4))
#y = np.arange(1,len(Xheader)+1)
#cmap = cm.coolwarm
#colors = [ cmap(x) for x in (k+1)/2 ]
#ax.hlines(y=y,xmin=0,xmax=k, color=colors, alpha=0.5, linewidth=12)
#ax.scatter(k,y,s=14**2,marker='o',c=k,cmap=cmap, vmin=-1,vmax=1,alpha=0.9)
#ax.set_xlabel('Components of $\mathbf{v}_b\mathbf{w}$', fontweight='black', color = '#333F4B')
#ax.set_ylabel('')
#ax.tick_params(axis='y', which='both',length=0)
#plt.yticks(y, Xheader)
#ax.set_yticks(np.append(y-0.5,np.max(y)+0.5), minor=True)
#ax.grid(axis='y',color='#333F4B', alpha=0.4,linestyle='-',which='minor',linewidth=1.5)
#ax.spines['top'].set_color('none')
#ax.spines['right'].set_color('none')
#ax.spines['left'].set_smart_bounds(True)
##ax.spines['bottom'].set_smart_bounds(True)
#ax.spines['bottom'].set_position(('axes', -0.04)) 
#ax.spines['left'].set_position(('axes', 0.015))
#ax.vlines(0,0.5,np.max(y)+0.5,color='#333F4B', alpha=0.4,linestyle='--',linewidth=1.5)
#ax.set_xlim([-1,1])

print('baseline y1 and y2:')
print(y1baseline)
print(y2baseline)

print('Sobol indices in y2 reduced dimension space:')
print(subpoly2.get_sobol_indices(order=1))

#dlist = [7,36,43,84,86,106,112,104,125,24,93,51,79,121,25,98,48,92,67,120,9,27,30,76,4,27,30,71,11]
#for d in range(128):#dlist:
#    if (X[d,4] > 0.4 and X[d,5] > 0.4 and X[d,6] < 0.5):
#        print(d,X[d,:])
##    print(y1[d])
##    print(y2[d])

plt.show()



