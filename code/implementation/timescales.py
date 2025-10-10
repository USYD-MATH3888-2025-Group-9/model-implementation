
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.differentiate import jacobian
from scipy.linalg import eig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import time
import itertools
import warnings
import copy #because I need it to clone the parameter instances
import concurrent.futures

from model_and_parameters import *

def timescale_phase_plots():
    params = Parameters()
    initial = {1:[40,0.1,0.1],2:[0.1,0.1],3:[0.1,0.1]}
    for ts in [1,2,3]:
        t_span = [0, 1200]
        numsteps = (t_span[1] - t_span[0]) * stepmul
        t_eval = np.linspace(t_span[0], t_span[1], numsteps)
        this_ts_cs = lambda t,x,p: ts_connor_stevens(t,x,p,ts)
        sol = solve_ivp(this_ts_cs, t_span, initial[ts], args=(params,), dense_output=True, t_eval=t_eval, method='RK45')
        if ts == 1:
            fig = plt.figure(figsize=(6,6))
            ax = fig.add_subplot(projection='3d')
            ax.plot(sol.y[0],sol.y[1],sol.y[2],color="blue",label="Path")
            ax.plot(sol.y[0][0],sol.y[1][0],sol.y[2][0],'go',label="Start")
            ax.plot(sol.y[0][-1],sol.y[1][-1],sol.y[2][-1],'ro',label="End")
            ax.set_title('Phase space: ' + pretty_names(0) + ' and ' + pretty_names(2) + " and " + pretty_names(3))
            ax.set_xlabel(pretty_names(0))
            ax.set_ylabel(pretty_names(2))
            ax.set_zlabel(pretty_names(3))
            ax.grid(True)
            ax.legend()
        elif ts == 2:
            fig = plt.figure(figsize=(12,6))
            ax = fig.add_subplot()
            ax.plot(sol.y[0],sol.y[1])
            ax.plot(sol.y[0][0],sol.y[1][0],'go',label="Start")
            ax.plot(sol.y[0][-1],sol.y[1][-1],'ro',label="End")
            ax.set_title('Phase space: ' + pretty_names(4) + ' and ' + pretty_names(5))
            ax.set_xlabel(pretty_names(4))
            ax.set_ylabel(pretty_names(5))
            ax.grid(True)
            ax.legend()
        elif ts == 3:
            fig = plt.figure(figsize=(12,6))
            ax = fig.add_subplot()
            ax.plot(sol.y[0],sol.y[1])
            ax.plot(sol.y[0][0],sol.y[1][0],'go',label="Start")
            ax.plot(sol.y[0][-1],sol.y[1][-1],'ro',label="End")
            ax.set_title('Phase space: ' + pretty_names(1) + ' and ' + pretty_names(6))
            ax.set_xlabel(pretty_names(1))
            ax.set_ylabel(pretty_names(6))
            ax.grid(True)
            ax.legend()
        plt.show()

def bifurcate_2d_timescales(ts,range_var1,range_var2,range_bifurcate,bifurcate_index,params): #index in [a2,a3,a4,b2,b3,b4]
    resolution = range_var1.shape[0]
    if resolution != range_var2.shape[0] or resolution != range_bifurcate.shape[0]:
        raise ValueError("The input arrays do not have the same shape/resolution: " \
                         + str(resolution) + ' (var 1), ' + str(range_var2.shape[0]) + ' (var 2), '\
                         + str(range_bifurcate.shape[0]) + '(bifurcate)')
    this_cs = lambda t,x,p: ts_connor_stevens(t,x,p,ts)
    eigvals_x = []
    var1grid, var2grid = np.meshgrid(range_var1,range_var2)
    vargrid = np.zeros_like(var1grid).tolist()
    for i in range(len(var1grid)):
        for j in range(len(var2grid)):
            vargrid[i][j] = [var1grid[i][j],var2grid[i][j]]
    vargrid = np.array(vargrid).T
    print(vargrid.shape)
    for x in range_bifurcate:
        params_mod = copy.deepcopy(params)
        params_mod.frozen_vars[bifurcate_index] = x
        jacobian_v = jacobian(lambda v: this_cs(0.1,v,params_mod),vargrid).df
        eigvals_v = []
        for vi in range(len(jacobian_v[0][0])):
            eigvals_vj = []
            for vj in range(len(jacobian_v[0][0])):
                matrix = jacobian_v[:,:,vi,vj]
                m_eigvals = eig(matrix,right=True)[0][0]
                eigvals_vj.append(m_eigvals)
            eigvals_v.append(eigvals_vj)
        eigvals_x.append(np.array(eigvals_v))
    bgrid, v1grid, v2grid = np.meshgrid(range_bifurcate,range_var1,range_var2)
    d1grid = np.zeros_like(bgrid)
    d2grid = np.zeros_like(bgrid)
    dbgrid = np.zeros_like(bgrid)
    for i in range(len(bgrid)):
        for j in range(len(bgrid[i])):
            for k in range(len(bgrid[i][j])):
                params_mod = copy.deepcopy(params)
                params_mod.frozen_vars[bifurcate_index] = bgrid[i][j][k]
                dgrid = this_cs(0.1,[v1grid[i][j][k],v2grid[i][j][k]],params_mod)
                d1grid[i][j][k] = dgrid[0]
                d2grid[i][j][k] = dgrid[1]
                full_vector = copy.deepcopy(params_mod.frozen_vars)
                full_vector.insert(0,0)
                if ts == 2:
                    full_vector[3] = v1grid[i][j][k]
                    full_vector[4] = v2grid[i][j][k]
                elif ts == 3:
                    full_vector[0] = v1grid[i][j][k]
                    full_vector[5] = v2grid[i][j][k]
                full_vector[bifurcate_index] = bgrid[i][j][k]
                dbgrid = connor_stevens(0,full_vector,params)[bifurcate_index + 1]
                full_vector = []
    final_x = np.array(eigvals_x).T
    print(final_x.shape)
    #display tolerance
    tol = 5e-5
    dzgrid = d1grid + d2grid #both zero is the only thing that matters
    red_dgrid = copy.deepcopy(d1grid)
    black_dgrid = copy.deepcopy(d1grid)
    signalgrid = np.zeros_like(d1grid)
    imag_signalgrid = np.zeros_like(d1grid)
    for i in range(len(dzgrid)):
        for j in range(len(dzgrid[i])):
            for k in range(len(dzgrid[i][j])):
                if final_x[i][j][k].imag != 0:
                    imag_signalgrid[i][j][k] = 1
                if final_x[i][j][k].real > 0:
                    red_dgrid[i][j][k] = np.nan #dummy
                    signalgrid[i][j][k] = 1
                    #black_dzgrid maintained
                elif final_x[i][j][k].real < 0:
                    black_dgrid[i][j][k] = np.nan #dummy
                    signalgrid[i][j][k] = -1
                    #red_dzgrid maintained
                else: #it was zero
                    signalgrid[i][j][k] = 0
    
    lp_points = []
    ah_points = []
    lp_in_grid = []
    ah_in_grid = []
    #now, adjacency check for signalgrid
    for i in range(len(dzgrid)):
        for j in range(len(dzgrid[i])):
            for k in range(len(dzgrid[i][j])):
                if abs(dzgrid[i][j][k]) > tol: continue
                checkpoints = [
                    signalgrid[i - 1 if i > 0 else i][j][k],
                    signalgrid[i + 1 if i < resolution - 1 else i][j][k],
                    signalgrid[i][j + 1 if j < resolution - 1 else j][k],
                    signalgrid[i][j - 1 if j > 0 else j][k],
                    signalgrid[i][j][k - 1 if k > 0 else k],
                    signalgrid[i][j][k + 1 if k < resolution - 1 else k]
                ]
                for p in checkpoints:
                    if signalgrid[i][j][k] != p:
                        if imag_signalgrid[i][j][k] == 1:
                            ah_points.append([bgrid[i][j][k],v1grid[i][j][k],v2grid[i][j][k]])
                            ah_in_grid.append([i,j,k])
                        else:
                            lp_points.append([bgrid[i][j][k],v1grid[i][j][k],v2grid[i][j][k]])
                            lp_in_grid.append([i,j,k])
                        break
    print(lp_points,ah_points)
    #graphing
    plt.figure(figsize=(12,6))
    ax = plt.axes(projection='3d')
    cr = ax.contour(bgrid,v1grid,v2grid,red_dgrid,colors='red',levels=[0])
    cb = ax.contour(bgrid,v1grid,v2grid,black_dgrid,colors='black',levels=[0])
    cx = ax.contour(bgrid,v1grid,v2grid,dbgrid,colors='purple',levels=[0])
    for p in lp_points:
        ax.plot(p[0],p[1],'go',label="LP Bifurcation at " + str(p[0]) + ", " + str(p[1]) + ', ' + str(p[2]))
    
    for p in ah_points:
        ax.plot(p[0],p[1],'bo',label="AH Bifurcation at " + str(p[0]) + ", " + str(p[1]) + ', ' + str(p[2]))
    
    plt.title('Bifurcation Diagram')
    plt.xlabel(pretty_names(bifurcate_index))
    if ts == 2:
        plt.ylabel('b2')
        plt.zlabel('b3')
    if ts == 3:
        plt.ylabel('a2')
        plt.zlabel('b4')
    plt.grid(True)
    plt.legend()
    plt.show()
