import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
from scipy.differentiate import jacobian
from scipy.linalg import eig
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines
import matplotlib.patches as mpatch
import matplotlib.animation as animate
import random
import time
import itertools
import warnings
import copy 
import concurrent.futures
import pickle
import sys
from collections import OrderedDict
from matplotlib import ticker
from scipy.differentiate import hessian

# This code is a modified version of the connor_stevens.py file used for the codimension 3 bifurcation analysis
# and the condition number calculation. The Bifurcation class has been expanded for multithreading.
# The plotting code is also here.

from connor_stevens import *

class Bifurcator:
    mod = None # modifier, converts parameter type to a new one with required value
    base = Parameters()
    model = connor_stevens
    p_range = np.linspace(0,100,100)
    threaded = False
    task_name = None
    this_j = None
    @staticmethod
    def _steady_states_task(new,i):
        steady,confidence = find_steady_states(param=new,tolerance_max=50)
        confidence_sorted = confidence.tolist()
        confidence_sorted.sort(reverse=True)
        confidence_finder = {}
        for j in range(len(steady)):
            confidence_finder[steady[j][0][0]] = confidence[j]
        steady.sort(reverse=True,key=lambda x: confidence_finder[x[0][0]])
        #print(steady,confidence_sorted)
        return [steady,confidence_sorted,i]
    
    def steady_states(self,tolerance_max=50):
        '''
        Compute steady states for this bifurcation setup
        '''
        if self.threaded:
            self.task_name = '_steady_states_task'
            return self.generate_modified_threaded()
        else:
            return self.generate_modified(lambda new,i: _steady_states_task(new,i))
    
    @staticmethod
    def _eigenvalue_dance_task(new,i,var_at_new):
        out = []
        for v in var_at_new:
            jacob = jacobian(lambda x: connor_stevens(0,x,new),v).df
            eig_data = eig(jacob,right=True)
            evectors = eig_data[1]
            evals = eig_data[0]
            #if verbose: print("eigenvalues",evals)
            out.append(np.array(evals))
        return [out,i]
    
    def eigenvalue_dance(self,var_range,sort_v_index):
        '''
        Find and trace the eigenvalues of the Jacobian for the value
        of the variables in var_range.
        '''
        evals_list = []
        if self.threaded:
            self.task_name = '_eigenvalue_dance_task'
            evals_list = self.generate_modified_threaded()
        else :
            evals_list = self.generate_modified(lambda new,i: Bifurcator._eigenvalue_dance_task(new,i,var_range[i]))
        #find the spot with the most channels
        max_channels = 1
        for i in evals_list:
            if len(i[0]) > max_channels:
                max_channels = len(i[0])
        #replicate until every list has the same number of channels (the max)
        for i in range(len(evals_list)):
            while len(evals_list[i][0]) < max_channels:
                evals_list[i][0].append(evals_list[i][0][-1]) #copies of the last one
            #print("HERE")
            key_list = list(var_range.keys())
            var_values_by_evals = {str(evals_list[i][0][x]):var_range[key_list[x]][0][sort_v_index] for x in range(len(evals_list[i][0]))}
            #print(evals_list[i][0])
            evals_list[i][0].sort(key=lambda x: var_values_by_evals[str(x)])
            #print(evals_list[i][0])
        #now fix the ordering channel-by-channel
        evals_by_channel = {}
        for ch in range(max_channels):
            second = False
            last_evals = np.array([])
            last2_evals = np.array([])
            final_evals = []
            for ev in evals_list:
                e = ev[0][ch]
                #print(e,last_evals,last2_evals)
                fixed_e = unscrambled(e,last_evals,last2_evals) if len(last_evals) != 0 and len(last2_evals) != 0 else e
                last2_evals = last_evals
                last_evals = fixed_e
                final_evals.append([fixed_e,ev[1]])
            evals_by_channel[ch] = final_evals
        return evals_by_channel

    def generate_modified(self,task):
        '''
        Computes and returns the results of an arbitrary task
        over the given parameter set, without threading.

        task: (Parameters,p) -> Any for p in p_range
        '''
        out_list = []
        counter = 0
        for i in self.p_range:
            #if verbose: print("count",counter)
            new_entry = copy.deepcopy(self.base)
            new_entry = self.mod(new_entry,i)
            #if verbose: print("running",self.p_range[counter])
            out_list.append(task(new_entry,i))
            counter += 1
        return out_list

    def generate_modified_threaded(self):
        '''
        Computes and returns the results of an arbitrary task
        over the given parameter set, with threading.

        task: (Parameters,p) -> Any for p in p_range
        '''
        out_list = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            results = [executor.submit(self._generate_modified_task,c) for c in range(len(self.p_range))]
            for r in concurrent.futures.as_completed(results):
                out_list.append(r.result())
        return out_list
    
    def _generate_modified_task(self,counter):
        i = self.p_range[counter]
        #if verbose: print("count",counter)
        new_entry = copy.deepcopy(self.base)
        if self.this_j == None:
            new_entry = self.mod(new_entry,i)
        else:
            new_entry = self.mod(new_entry,i,self.this_j)
        print("running",i)
        match self.task_name:
            case '_steady_states_task':
                return self._steady_states_task(new_entry,i)
            case '_eigenvalue_dance_task':
                return self._eigenvalue_dance_task(new_entry,i)
        #return self.current_task(new_entry,i)
    
    def __init__(self,mod,base,p_range,threaded):
        self.mod = mod
        self.base = base
        self.p_range = p_range
        self.threaded = threaded

def colours(index):
    match index:
        case 0:
            return "red"
        case 1:
            return "orange"
        case 2:
            return "yellow"
        case 3:
            return "green"
        case 4:
            return "blue"
        case 5:
            return "purple"
        case 6:
            return "black"

def perform_bifurcation(test,tolerance_max=50):
    bifurcate_results = test.steady_states(tolerance_max=tolerance_max)
    return bifurcate_results

bifurcation_coordinates = {
    'v_2_pitchfork':[-3,0.76,0.92,0.87,0.36,0.005,0.055],
    'v_3_pitchfork':[-27,0.965,0.32,0.737,0,0.336,0.156]}
bif_param_values = {
    'v_2_pitchfork':{'v_2':8,'v_3':-45},
    'v_3_pitchfork':{'v_2':-60,'v_3':61}
}

def condition_number(v,params,bifurcation_coordinates):
    matrix_name = 'hessian'
    spot = copy.deepcopy(bifurcation_coordinates)
    spot[0] = v
    ntx = None
    match matrix_name:
        case 'jacobian':
            mtx = jacobian(lambda x: connor_stevens(0,x,params),spot).df
        case 'hessian':
            mtx = hessian(lambda x: connor_stevens(0,x,params),spot).ddf
        case 'powell':
            jac = jacobian(lambda x: connor_stevens(0,x,params),spot).df
            mtx = np.linalg.inv(jac.T @ jac) @ jac.T
    mtx_inverse = np.linalg.inv(mtx)
    evals = np.abs(eig(mtx,right=True)[1])
    max_evals = np.max(evals)
    min_evals = np.min(evals)
    inv_evals = np.abs(eig(mtx_inverse,right=True)[1])
    max_inv = np.max(evals)
    min_inv = np.min(evals)
    condition = (max_evals * max_inv) / (min_evals * min_inv)
    return np.log10(condition)

def plot_bifurcation(bifurcate_results,var_index,param_range=None,min_confidence_plot=None,nearby_tolerance=None,vertical=None):
    doing_cts = False
    worst_case = True
    
    if isinstance(param_range,np.ndarray):
        pgrid,vgrid = np.meshgrid(param_range,var_values)
        steadygrid = np.zeros_like(vgrid) - 1
        print(pgrid.shape,vgrid.shape)
        var_values = []
        for b in bifurcate_results:
            for a in range(len(b[0])):
                var_values.append(b[0][a][0][var_index])
        var_values = np.array(sorted(var_values))
        var_indexes = {var_values[v]:v for v in range(len(var_values))}
        param_indexes = {param_range[p]:p for p in range(len(param_range))}
        print(var_indexes,'\n',var_values)
        doing_cts = True
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot()
    #trying to find the bifurcation spot
    #calculated by trial and error
    bifurcation_coordinates = [-3,0.76,0.92,0.87,0.36,0.005,0.055]
    bif_param_value = 8
    ax.scatter([bif_param_value],[bifurcation_coordinates[var_index]],s=200,marker='3',color='#00ff00',zorder=10)
    positions = {}
    for b in bifurcate_results:
        for a in range(len(b[0])):
            if b[1][a] >= 0.4 if not worst_case else True:
                ax.plot(b[2],b[0][a][0][var_index],"o",color=(b[1][a],0,0))
                to_add = b[0][a][0]
                positions[(b[2],b[0][a][0][var_index])] = to_add
            if doing_cts:
                steadygrid[var_indexes[b[0][a][0][var_index]]][param_indexes[b[2]]] = b[1][a]
    
    p_range = np.linspace(-40,80,100)
    v_range = np.linspace(-100,100,100)
    other_range = np.linspace(0,1,100)
    params = Parameters()
    pgrid,vgrid = np.meshgrid(p_range,v_range) if var_index == 0 else np.meshgrid(p_range,other_range)
    egrid = np.zeros_like(pgrid)
    if worst_case:
        #compute condition number of the problem
        pcount = 0
        for p in p_range:
            params.v_j[1] = p
            for c in range(len(v_range)):
                distances = {pos:((p - pos[0]) ** 2 + (v_range[c] - pos[1]) ** 2) ** 0.5 for pos in positions.keys()}
                total_dist = np.sum(np.array(list(distances.values())))
                weighted_distances = {pos:distances[pos] / float(total_dist) for pos in distances.keys()}
                totals = []
                for i in [0,1,2,3,4,5,6]:
                    total = 0
                    count = 0
                    for pos in distances.keys():
                        total += weighted_distances[pos] * positions[pos][i]
                        count += 1
                    totals.append(total / count)
                target_coordinates = [v_range[c] if i == var_index else totals[i] for i in [0,1,2,3,4,5,6]]
                egrid[pcount][c] = condition_number(v_range[c] if var_index == 0 else other_range[c],params,bifurcation_coordinates)
                if egrid[pcount][c] > 10 and var_index != 0:
                    egrid[pcount][c] = 10
            pcount += 1
    print(egrid[:,0],egrid[0,:])
    contours = ax.contourf(pgrid,vgrid,egrid,levels=100,cmap='winter')
    fig.colorbar(contours,label='$Log_{10}$ Condition Number')
    if var_index == 0:
       ax.set_ylim([-75,75])
    if var_index == 1:
       ax.set_xlim([-37,80])
    ax.set_ylabel(pretty_names(var_index))
    ax.set_xlabel("$V_2$ Potassium Nernst Potential")
    ax.set_title("$V_2$ vs " + pretty_names(var_index) + " - Kyle George, SID 520417425")
    ax.grid(True)
    short_names = ['$V(t)$','$a_2$','$a_3$','$a_4$','$b_2$','$b_3$','$b_4$']
    sneaky1_line = mlines.Line2D([], [], color="#00000000", marker='3', markerfacecolor="#00ff00",markeredgecolor='#00ff00',markersize=12,label='Bifurcation: $V_2$ = 8mV,' + short_names[var_index] + ' = ' + str(bifurcation_coordinates[var_index]) + ('mV' if var_index == 0 else ''))
    sneaky2_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#000000",markeredgecolor='#630000',label='Steady state, 0.4 confidence')
    sneaky3_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#ff0000",markeredgecolor='#ff0000',label='Steady state, 1.0 confidence')
    plt.legend(handles=[sneaky1_line,sneaky2_line,sneaky3_line])
    plt.show()
    if doing_cts:
        final_steadygrid = copy.deepcopy(steadygrid)
        for i in range(len(steadygrid)):
            for j in range(len(steadygrid[i])):
                this_value = steadygrid[i][j]
                #check for min_confidence
                final_steadygrid[i][j] = 1 if min_confidence_plot <= this_value else 0
                
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot()
        print(final_steadygrid)
        ppoints = []
        vpoints = []
        for i in range(len(final_steadygrid)):
            for j in range(len(final_steadygrid[i])):
                if final_steadygrid[i][j] == 1:
                    ppoints.append(param_range[j])
                    vpoints.append(var_values[i])
        for i in range(len(ppoints) - 1):
            #verify that the closest other point is near enough
            min_distance = -1
            for j in range(len(ppoints)):
                this_distance = (abs(ppoints[j] - ppoints[i]) ** 2 + abs(vpoints[j] - vpoints[i]) ** 2) ** (1/2)
                if min_distance == -1 or (min_distance > this_distance and this_distance > 0):
                    min_distance = this_distance
            print(min_distance)
            if abs(ppoints[i] - ppoints[i + 1]) > vertical and min_distance <= nearby_tolerance:
                ax.plot([ppoints[i],ppoints[i + 1]],[vpoints[i],vpoints[i + 1]],'r-')
        ax.set_ylabel(pretty_names(var_index))
        ax.set_xlabel("$V_2$ Potassium Nernst Potential (mV)")
        ax.set_title("$V_2$ vs " + pretty_names(var_index) + " - Kyle George, SID 520417425")
        ax.grid(True)
        ax.legend()
        plt.show()
    
class MissingStateError(Exception):
     def __init__(self, message="This is a custom error."):
        self.message = message
        super().__init__(self.message)


def eigenvalue_plot(test,ssr,sort_var_index,min_confidence_plot=0.5,continuous_fake=False):
    steady_states = {}
    last_index = None
    max_channels = 1
    for i in ssr:
        if len(i[0]) != 0:
            will_include = []
            for x in range(len(i[0])):
                if i[1][x] >= min_confidence_plot:
                    will_include.append(i[0][x][0])
            if len(will_include) > max_channels:
                max_channels = len(will_include)
    for i in ssr:
        if len(i[0]) != 0:
            will_include = []
            for x in range(len(i[0])):
                if i[1][x] >= min_confidence_plot:
                    will_include.append(i[0][x][0])
            if len(will_include) < max_channels:
                if continuous_fake:
                    if last_index != None:
                        for ch in range(len(will_include),len(steady_states[last_index])):
                            will_include.append(steady_states[last_index][ch])
                        #print("Channel: Fake value used for",i[2],":",will_include)
                    else:
                        #print("Channel: Bad continuous fake used for",i[2])
                        while len(will_include) < max_channels:
                            will_include.append(will_include[0])
                else:
                    raise MissingStateError(message="Missing state info for value " + str(i) + " channel " + str(ch))
            steady_states[i[2]] = will_include
            last_index = i[2]
        else:
            if continuous_fake:
                steady_states[i[2]] = steady_states[last_index]
                print("State: Fake value used for",i[2],":",steady_states[i[2]])
            else:
                raise MissingStateError(message="Missing state info for value " + str(i))
    bfr2 = test.eigenvalue_dance(steady_states,sort_v_index=sort_var_index)
    for ch,bifurcate_results2 in bfr2.items():
        print("Channel",ch)
        gs = gridspec.GridSpec(2,2)
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(gs[0,0],projection="3d")
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[0,1])
        ax4 = fig.add_subplot(gs[1,1])
        for a in [0,1,2,3,4,5,6]:
            pointx = []
            pointy = []
            pointz = []
            for b in bifurcate_results2:
                pointx.append(b[0][a].real)
                pointy.append(b[0][a].imag)
                pointz.append(b[1])
            pointx = np.array(pointx)
            pointy = np.array(pointy)
            pointz = np.array(pointz)
            ax.plot(pointx,pointy,pointz,color=colours(a),label="Eigenvalue " + str(a))
            ax.plot(pointx[0],pointy[0],pointz[0],'o',color=colours(a),label="Start for " + str(a))
            ax2.plot(pointx,pointy,color=colours(a),label="Eigenvalue " + str(a))
            ax2.plot(pointx[0],pointy[0],'o',color=colours(a),label="Start for " + str(a))
            ax3.plot(pointx,pointz,color=colours(a),label="Eigenvalue " + str(a))
            ax3.plot(pointx[0],pointz[0],'o',color=colours(a),label="Start for " + str(a))
            ax4.plot(pointy,pointz,color=colours(a),label="Eigenvalue " + str(a))
            ax4.plot(pointy[0],pointz[0],'o',color=colours(a),label="Start for " + str(a))
        labels = ["Eigenvalue real","Eigenvalue imaginary","$V_2$ Potassium Nernst Potential (mV)"]
        ax.set_zlabel(labels[2])
        ax.set_ylabel(labels[1])
        ax.set_xlabel(labels[0])
        ax2.set_ylabel(labels[1])
        ax2.set_xlabel(labels[0])
        ax3.set_ylabel(labels[2])
        ax3.set_xlabel(labels[0])
        ax4.set_ylabel(labels[2])
        ax4.set_xlabel(labels[1])
        ax.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)
        ax.legend()
        plt.show()

def v_inf(t,x,p):
    a2,a3,a4,b2,b3,b4 = x
    c1 = p.g(2) * (a2 ** 2) * b2 + p.g(3) * (a3 ** 3) * b3 + p.g(4) * (a4 ** 4) * b4
    c2 = p.g(2) * (a2 ** 2) * b2 * p.v(2) + p.g(3) * (a3 ** 3) * b3 * p.v(3) + p.g(4) * (a4 ** 4) * b4 * p.v(4)
    return (p.I(t) - c2) / c1

def ts_connor_stevens(t,x,p,timescale):
    frozen = p.frozen_vars
    if timescale == 1: #fastest, v, a3, a4
        v,a3,a4 = x
        da3dt = p.ainf(3,v) - a3
        da4dt = p.ainf(4,v) - a4
        dvdt = p.I(t) - (a3 ** 3) * p.binf(3,v) * (v - p.v(3)) \
             - (a4 ** 4) * p.binf(4,v) * (v - p.v(4)) \
             - (p.ainf(2,v) ** 2) * p.binf(4,v) * (v - p.v(2))
        return [da3dt,da4dt,dvdt]
    elif timescale == 2: #middle, b2, b3
        b2,b3 = x
        v = v_inf(t,[frozen[0],frozen[1],frozen[2],b2,b3,frozen[5]],p)
        db2dt = p.binf(2,v) - b2
        db3dt = p.binf(3,v) - b3
        return [db2dt,db3dt]
    else: #slowest, b4, a2
        a2,b4 = x
        v = v_inf(t,[a2,frozen[1],frozen[2],frozen[3],frozen[4],b4],p)
        da2dt = p.ainf(2,v) - a2
        db4dt = p.binf(4,v) - b4
        return [da2dt,db4dt]

def timescale_phase_plots():
    params = Parameters()
    initial = {1:[40,0.1,0.1],2:[0.1,0.1],3:[0.1,0.1]}
    for ts in [1,2,3]:
        t_span = [0, 1200]
        t_eval = np.linspace(t_span[0], t_span[1], t_span[1] - t_span[0])
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

###########################
# Actually run the system #
###########################

#THIS IS REALLY IMPORTANT!!!
#The parameter to bifurcate on is changed here
def modifier(p,m):
    out = p
    out.v_j[1] = 20
    out.v_j[2] = -10
    out.v_j[3] = m
    return out

def modifier2(p,m):
    out = p
    out.v_j[2] = 45
    out.Iapp = Constant(m)
    return out

def modifier3(p,m,other):
    out = p
    out.v_j[1] = other
    out.v_j[2] = m
    return out

# basic_system_data()
#soln = basic_system_data()
#phase_planes(soln)
#test = Bifurcator(modifier,Parameters(),np.linspace(-35,35,50))
#steady_state_results = perform_bifurcation(test)
#eigenvalue_plot(test,steady_state_results,continuous_fake=True)

#Threaded system speed test
#This does NOT work for eigenvalue dance as it's not really needed
def full_system():
    starting = time.time()
    count = 0
    full_results = {}
    for j in np.linspace(-10,40,100):
        print('starting',j)
        new_data = True
        if count % 10 == 0 and count != 0:
            print('Resting')
            time.sleep(180)
            print('Restarting')
        param_range = np.linspace(-80,80,100)
        thread_start = time.time()
        global this_modifier
        this_modifier = lambda p,m: modifier3(p,m,j)
        test1 = Bifurcator(modifier3,Parameters(),param_range,True)
        test1.this_j = j
        steady_state_results1 = perform_bifurcation(test1,tolerance_max=100)
        threaded_time = time.time() - thread_start
        print(j,"took:",threaded_time,'estimated total',threaded_time * 100)
        if count != 0:
            with open('sr_full_3d.pickle','rb') as file:
                full_results = pickle.load(file)
        full_results[j] = steady_state_results1
        with open('sr_full_3d.pickle','wb') as file:
            pickle.dump(full_results,file)
        count += 1
    ending = time.time() - starting
    print("Finished in",ending,'s')
    
# the codimension 3 bifurcation is at:
# [v_2,v_3]=[11.0,0.00,-42]
# [v,a2,a3,a4,b2,b3,b4]=[-10,0.8,0.9,0.9,0.2,0.1,0.1]
def graph_3d():
    for var_index in [0,1,2,3,4,5,6]:
        #plot
        
        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot()
        results = {}
        with open('sr_full_3d.pickle','rb') as file:
            results = pickle.load(file)
        #swap the variables
        '''
        flipped_results = {}
        for k in results.keys():
            for b in results[k]:
                flipped_results[b[2]] = []
        for k in results.keys():
            for b in results[k]:
                new_b = [b[0],b[1],k]
                flipped_results[b[2]].append(new_b)
        '''
        #find and plot the appropriate values for this variable
        var_results = OrderedDict() #{var_value:[for each point: [v2,v3,confidence]]}
        scaler = 2
        for k in results.keys():
            for b in results[k]:
                for a in range(len(b[0])):
                    if b[1][a] >= 0.4:
                        rounded_var = round(b[0][a][0][var_index] / scaler) if var_index == 0 else round(b[0][a][0][var_index],2)
                        var_results[rounded_var * (scaler if var_index == 0 else 1)] = []
        for k in results.keys():
            for b in results[k]:
                for a in range(len(b[0])):
                    if b[1][a] >= 0.4:
                        rounded_var = round(b[0][a][0][var_index] / scaler) if var_index == 0 else round(b[0][a][0][var_index],2)
                        var_results[rounded_var * (scaler if var_index == 0 else 1)].append([b[2],k,b[1][a]])
        var_results = OrderedDict(sorted(var_results.items(),key=lambda x: x[0]))
        v_values = list(var_results.keys())
        print(v_values)
        minimum_v = min(v_values)
        maximum_v = max(v_values)
        plot_bits = []
        for k in var_results.keys():
            p2points = []
            p3points = []
            colours = []
            for b in var_results[k]:
                p2points.append(b[0])
                p3points.append(b[1])
                colours.append((b[2],0,0))
            this_plot = ax.scatter(p2points,p3points,c=colours)
            #vanity
            ax.set_ylabel("$V_3$ Sodium Nernst Potential (mV)")
            ax.set_xlabel("$V_2$ Potassium Nernst Potential (mV)")
            ax.set_title("$V_2$, $V_3$ vs " + pretty_names(var_index) + " - Kyle George")
            ax.grid(True)
            short_names = ['$V(t)$','$a_2$','$a_3$','$a_4$','$b_2$','$b_3$','$b_4$']
            #sneaky1_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#00000000",markeredgecolor='#00000000',markersize=12,label='Bifurcation: $V_2$ = 8mV,' + short_names[var_index] + ' = ' + str(bifurcation_coordinates[var_index]) + ('mV' if var_index == 0 else ''))
            sneaky2_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#000000",markeredgecolor='#630000',label='Steady state, 0.4 confidence')
            sneaky3_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#ff0000",markeredgecolor='#ff0000',label='Steady state, 1.0 confidence')
            sneaky1_line = mlines.Line2D([], [], color="#00000000", marker='o',markerfacecolor="#00000000",markeredgecolor='#00000000',markersize=12,label=f"{short_names[var_index]}$\\in[{minimum_v},{maximum_v}]$ animated")
            legend = ax.legend(handles=[sneaky2_line,sneaky3_line,sneaky1_line])
            plot_bits.append([this_plot])
        #ax.vlines(0,-50,50,color="green")
        ani = animate.ArtistAnimation(fig,plot_bits,interval=(1/10)*1000,repeat=True)
        ani.save('animation_v2v3_on_variable_' + short_names[var_index][1:-1] + '.mp4', writer=animate.FFMpegWriter(fps=10))
        #plt.show()
        '''
        plot_bits = []
        for k in results.keys():
            ppoints = []
            vpoints = []
            colours = []
            for b in results[k]:
                for a in range(len(b[0])):
                    if b[1][a] >= 0.4:
                        ppoints.append(b[2])
                        vpoints.append(b[0][a][0][var_index])
                        colours.append((b[1][a],0,0))
            this_plot = ax.scatter(ppoints,vpoints,c=colours)
            #vanity
            ax.set_ylabel(pretty_names(var_index))
            #ax.set_ylabel("$V_3$ Sodium Nernst Potential (mV)")
            ax.set_xlabel("$V_2$ Potassium Nernst Potential (mV)")
            ax.set_title("$V_2$, $V_3$ vs " + pretty_names(var_index) + " - Kyle George")
            ax.grid(True)
            short_names = ['$V(t)$','$a_2$','$a_3$','$a_4$','$b_2$','$b_3$','$b_4$']
            #sneaky1_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#00000000",markeredgecolor='#00000000',markersize=12,label='Bifurcation: $V_2$ = 8mV,' + short_names[var_index] + ' = ' + str(bifurcation_coordinates[var_index]) + ('mV' if var_index == 0 else ''))
            sneaky2_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#000000",markeredgecolor='#630000',label='Steady state, 0.4 confidence')
            sneaky3_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#ff0000",markeredgecolor='#ff0000',label='Steady state, 1.0 confidence')
            sneaky1_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#00000000",markeredgecolor='#00000000',markersize=12,label='$V_3=\\in[-10,27]$ animated')
            legend = ax.legend(handles=[sneaky2_line,sneaky3_line,sneaky1_line])
            plot_bits.append([this_plot])
        print(results.keys())
        #ax.vlines(0,-50,50,color="green")
        ani = animate.ArtistAnimation(fig,plot_bits,interval=(1/15)*1000,repeat=True)
        ani.save('animation_on_v3_modified_' + short_names[var_index][1:-1] + '.gif', writer=animate.PillowWriter(fps=15))
        #plt.show()
        '''
       
def final_graph_3d():
    for var_index in [0,1,2,3,4,5,6]:
        gs = gridspec.GridSpec(2,2)
        fig = plt.figure(figsize=(12,6))
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0])
        ax3 = fig.add_subplot(gs[0,1])
        axf = fig.add_subplot(gs[1,1],projection='3d')
        results = {}
        with open('sr_full_3d.pickle','rb') as file:
            results = pickle.load(file)
        plot_bits = []
        for k in results.keys():
            ppoints = []
            vpoints = []
            colours = []
            for b in results[k]:
                for a in range(len(b[0])):
                    if b[1][a] >= 0.4:
                        ppoints.append(b[2])
                        vpoints.append(b[0][a][0][var_index])
                        colours.append((b[1][a],0,0))
            this_plot = ax1.scatter(ppoints,vpoints,c=colours)
            #vanity
            ax1.set_ylabel(pretty_names(var_index))
            #ax.set_ylabel("$V_3$ Sodium Nernst Potential (mV)")
            ax1.set_xlabel("$V_2$ Potassium Nernst Potential (mV)")
            ax1.set_title("$V_2$, $V_3$ vs " + pretty_names(var_index) + " - Kyle George")
            ax1.grid(True)
            short_names = ['$V(t)$','$a_2$','$a_3$','$a_4$','$b_2$','$b_3$','$b_4$']
            #sneaky1_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#00000000",markeredgecolor='#00000000',markersize=12,label='Bifurcation: $V_2$ = 8mV,' + short_names[var_index] + ' = ' + str(bifurcation_coordinates[var_index]) + ('mV' if var_index == 0 else ''))
            sneaky2_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#000000",markeredgecolor='#630000',label='Steady state, 0.4 confidence')
            sneaky3_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#ff0000",markeredgecolor='#ff0000',label='Steady state, 1.0 confidence')
            sneaky1_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#00000000",markeredgecolor='#00000000',markersize=12,label='$V_3=\\in[-10,27]$ animated')
            plot_bits.append(this_plot)
        #ax.vlines(0,-50,50,color="green")
        flipped_results = {}
        for k in results.keys():
            for b in results[k]:
                flipped_results[b[2]] = []
        for k in results.keys():
            for b in results[k]:
                new_b = [b[0],b[1],k]
                flipped_results[b[2]].append(new_b)
        plot_bits2 = []
        for k in flipped_results.keys():
            ppoints = []
            vpoints = []
            colours = []
            for b in flipped_results[k]:
                for a in range(len(b[0])):
                    if b[1][a] >= 0.4:
                        ppoints.append(b[2])
                        vpoints.append(b[0][a][0][var_index])
                        colours.append((0,b[1][a],0))
            this_plot = ax2.scatter(ppoints,vpoints,c=colours)
            #vanity
            ax2.set_ylabel(pretty_names(var_index))
            #ax.set_ylabel("$V_3$ Sodium Nernst Potential (mV)")
            ax2.set_xlabel("$V_3$ Sodium Nernst Potential (mV)")
            ax2.grid(True)
            short_names = ['$V(t)$','$a_2$','$a_3$','$a_4$','$b_2$','$b_3$','$b_4$']
            #sneaky1_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#00000000",markeredgecolor='#00000000',markersize=12,label='Bifurcation: $V_2$ = 8mV,' + short_names[var_index] + ' = ' + str(bifurcation_coordinates[var_index]) + ('mV' if var_index == 0 else ''))
            sneaky2_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#000000",markeredgecolor='#006300',label='Steady state, 0.4 confidence')
            sneaky3_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#00ff00",markeredgecolor='#00ff00',label='Steady state, 1.0 confidence')
            sneaky1_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#00000000",markeredgecolor='#00000000',markersize=12,label='$V_2=\\in[-80,80]$ animated')
            plot_bits2.append(this_plot)
        #ax.vlines(0,-50,50,color="green")
        var_results = OrderedDict() #{var_value:[for each point: [v2,v3,confidence]]}
        scaler = 2
        for k in results.keys():
            for b in results[k]:
                for a in range(len(b[0])):
                    if b[1][a] >= 0.4:
                        rounded_var = round(b[0][a][0][var_index] / scaler) if var_index == 0 else round(b[0][a][0][var_index],2)
                        var_results[rounded_var * (scaler if var_index == 0 else 1)] = []
        for k in results.keys():
            for b in results[k]:
                for a in range(len(b[0])):
                    if b[1][a] >= 0.4:
                        rounded_var = round(b[0][a][0][var_index] / scaler) if var_index == 0 else round(b[0][a][0][var_index],2)
                        var_results[rounded_var * (scaler if var_index == 0 else 1)].append([b[2],k,b[1][a]])
        var_results = OrderedDict(sorted(var_results.items(),key=lambda x: x[0]))
        v_values = list(var_results.keys())
        print(v_values)
        minimum_v = min(v_values)
        maximum_v = max(v_values)
        plot_bits3 = []
        for k in var_results.keys():
            p2points = []
            p3points = []
            colours = []
            for b in var_results[k]:
                p2points.append(b[0])
                p3points.append(b[1])
                colours.append((0,0,b[2]))
            this_plot = ax3.scatter(p2points,p3points,c=colours)
            #vanity
            ax3.set_ylabel("$V_3$ Sodium Nernst Potential (mV)")
            ax3.set_xlabel("$V_2$ Potassium Nernst Potential (mV)")
            ax3.grid(True)
            short_names = ['$V(t)$','$a_2$','$a_3$','$a_4$','$b_2$','$b_3$','$b_4$']
            #sneaky1_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#00000000",markeredgecolor='#00000000',markersize=12,label='Bifurcation: $V_2$ = 8mV,' + short_names[var_index] + ' = ' + str(bifurcation_coordinates[var_index]) + ('mV' if var_index == 0 else ''))
            sneaky2_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#000000",markeredgecolor='#000063',label='Steady state, 0.4 confidence')
            sneaky3_line = mlines.Line2D([], [], color="#00000000", marker='o', markerfacecolor="#0000ff",markeredgecolor='#0000ff',label='Steady state, 1.0 confidence')
            sneaky1_line = mlines.Line2D([], [], color="#00000000", marker='o',markerfacecolor="#00000000",markeredgecolor='#00000000',markersize=12,label=f"{short_names[var_index]}$\\in[{minimum_v},{maximum_v}]$ animated")
            plot_bits3.append(this_plot)
        print(len(flipped_results.keys()),len(results.keys()),len(var_results.keys()))
        v2range = np.linspace(-80,80,len(flipped_results.keys()))
        v3range = np.linspace(-10,27,len(results.keys()))
        var_range = np.linspace(-100,70,len(var_results.keys())) if var_index == 0 else np.linspace(0,1,len(var_results.keys()))
        plot_bits4 = []
        plot_bits5 = []
        plot_bits6 = []
        plot_bits7 = []
        for i in range(101):
            #generate the sliding planes
            v2v3g, v3v2g = np.meshgrid(v2range,v3range)
            varv2g = np.full_like(v2v3g, var_range[i if i < len(var_results.keys()) else len(var_results.keys()) - 1])
            plot_bits4.append(axf.plot_surface(v2v3g,v3v2g,varv2g,alpha=0.7,color='#0000ff'))
            
            v2varg, varv2g = np.meshgrid(v2range,var_range)
            varv3g = np.full_like(v2varg, v3range[i if i < len(results.keys()) else len(results.keys()) - 1])
            plot_bits5.append(axf.plot_surface(v2varg,varv3g,varv2g,alpha=0.7,color='#ff0000'))
            
            v3varg, varv3g = np.meshgrid(v3range,var_range)
            varv2g = np.full_like(v3varg, v2range[i if i < len(flipped_results.keys()) else len(flipped_results.keys()) - 1])
            plot_bits6.append(axf.plot_surface(varv2g,v3varg,varv3g,alpha=0.7,color='#00ff00'))
            title_text = f"$V={var_range[i if i < len(var_results.keys()) else len(var_results.keys()) - 1]:.2f}$,$V_2={v2range[i if i < len(flipped_results.keys()) else len(flipped_results.keys()) - 1]:.2f}$,$V_3={v3range[i if i < len(results.keys()) else len(results.keys()) - 1]:.2f}$"
            
            plot_bits7.append(ax3.text(0.5, 1.01, title_text, horizontalalignment='center', verticalalignment='bottom', transform=ax3.transAxes))
            
        axf.set_xlabel('$V_2$ [-80,80]')
        axf.set_ylabel('$V_3$ [-10,27]')
        axf.set_zlabel(short_names[var_index] + ('[-100,70]' if var_index == 0 else '[0,1]'))
        #we want a 10.1s animation at 10fps, so 101 frames
        final_plot_bits = []
        for i in range(101):
            final_plot_bits.append([
                plot_bits[i if i < len(plot_bits) else len(plot_bits) - 1],
                plot_bits2[i if i < len(plot_bits2) else len(plot_bits2) - 1],
                plot_bits3[i if i < len(plot_bits3) else len(plot_bits3) - 1],
                plot_bits4[i],
                plot_bits5[i],
                plot_bits6[i],
                plot_bits7[i]
            ])
        ani = animate.ArtistAnimation(fig,final_plot_bits,interval=(1/15)*1000,repeat=True)
        ani.save('triple_animation_' + short_names[var_index][1:-1] + '.gif', writer=animate.PillowWriter(fps=15))
        #plt.show()

from scipy.spatial import Voronoi, voronoi_plot_2d
def main():
    #full_system()
    #final_graph_3d()
    #return
    
    #test_p = Parameters()
    #test_p.v_j[1] = 2
    #ssr = find_steady_states(verbose=True,param=test_p,tolerance_max=50)
    #return
    
    new_data = False
    param_range = np.linspace(-40,80,250)
    thread_start = time.time()
    test1 = Bifurcator(modifier2,Parameters(),param_range,True) #this True/False controls if the threading is on
    if new_data:
        steady_state_results1 = perform_bifurcation(test1,tolerance_max=50)
        threaded_time = time.time() - thread_start
        print("1: Threaded took:",threaded_time)
        with open('sr_iapp_m_highres.pickle','wb') as file:
            pickle.dump(steady_state_results1,file)
    with open('steady_results.pickle','rb') as file:
        steady_state_results1 = pickle.load(file)
        for index in [0]:
            #plot_bifurcation(steady_state_results1,index,param_range,0.4,4 if index == 0 else 1,1e-3)
            plot_bifurcation(steady_state_results1,index)
        test1.threaded = False
        #eigenvalue_plot(test1,steady_state_results1,0,continuous_fake=True)
    #un_start = time.time()
    #test = Bifurcator(modifier,Parameters(),np.linspace(-80,80,50),False)
    #steady_state_results = perform_bifurcation(test,tolerance_max=50)
    #plot_bifurcation(steady_state_results)
    #unthreaded_time = time.time() - un_start
    #print("Unthreaded took:",unthreaded_time)

    #This currently functions, but fails to graph it.
    #bifurcate_2d_timescales(2,np.linspace(0,1,100),np.linspace(0,1,100),np.linspace(0,1,100),1,Parameters())
    #timescale_phase_plots()

if __name__ == "__main__":
    main()
