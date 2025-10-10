#!/usr/bin/python3

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

# --- Setup
def linear(x,m,b):
    return m * x + b

class CappedLinear:
    grad = 0
    yint = 0
    lower = 1
    upper = 0
    minimum = 0
    maximum = 1
    reverse = False
    def run(self,x):
        if isinstance(x,np.ndarray):
            out = linear(x,self.grad,self.yint)
            final_shape = out.shape
            out = np.reshape(out,(-1))
            new_x = np.reshape(x,(-1))
            for i in range(len(out)):
                if new_x[i] < self.upper if self.reverse else new_x[i] > self.upper:
                    out[i] = self.maximum if self.reverse else self.minimum
                elif new_x[i] > self.lower if self.reverse else new_x[i] < self.lower:
                    out[i] = self.minimum if self.reverse else self.maximum
            return np.reshape(out,final_shape)
        else:
            out = linear(x,self.grad,self.yint)
            if x < self.upper if self.reverse else x > self.upper:
                return self.maximum if self.reverse else self.minimum
            elif x > self.lower if self.reverse else x < self.lower:
                return self.minimum if self.reverse else self.maximum
            else:
                return out
    def __init__(self,grad,yint,lower,upper,minimum,maximum,reverse):
        self.grad = grad
        self.yint = yint
        self.lower = lower
        self.upper = upper
        self.minimum = minimum
        self.maximum = maximum
        self.reverse = reverse

def sigmoid(x,a,b,c,d):
    return a / (b + np.exp((c - x) / d))

class Sigmoidal:
    a = 0
    b = 0
    c = 0
    d = 0
    yshift = 0
    xshift = 0
    def run(self,x):
        return sigmoid(x + self.xshift,self.a,self.b,self.c,self.d) + self.yshift
    def __init__(self,a,b,c,d,yshift=0,xshift=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.yshift = yshift
        self.xshift = xshift

class PowerSigmoidal:
    a = 0
    b = 0
    c = 0
    d = 0
    power = 1 / 2
    xshift = 0
    yshift = 0
    def run(self,x):
        return np.power(sigmoid(x + self.xshift,self.a,self.b,self.c,self.d),self.power) + self.yshift
    def __init__(self,a,b,c,d,power,xshift=0,yshift=0):
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.power = power
        self.xshift = xshift
        self.yshift = yshift

class StepFunction:
    #type: list<pairs:start,value>, must be ordered.
    steps = []
    default = 0
    def run(self,x):
        out = self.default
        for start,value in self.steps:
            if x >= start:
                out = value
        return out
    def __init__(self,steps,default):
        self.steps = steps
        self.default = default

class Constant:
    value = 0
    def run(self,x):
        return self.value
    def __init__(self,value):
        self.value = value

# --- Physiological Parameters ---
class Parameters:
    cm = 14 # trial, 14pF
    disabled = [0,1,1,1,1] #blank,blank,2,3,4

    v_j = [-40,-60,-45,-63]
    g_j = [0.049,10,21,10]
    atau_j = [Constant(1),
            CappedLinear(
                grad=(300-60)/(-50),
                yint=(((300-60) / (-50)) * 40 + 300),
                lower=-40,
                upper=10,
                maximum=300,
                minimum=60,
                reverse=False),
            Sigmoidal(
                a=0.6335,
                b=0.05426,
                c=-1.932,
                d=-4.650,
                yshift=1
            ),
           Constant(12)]
    btau_j = [
        Constant(1),
        Constant(50),
        Sigmoidal(
            a=0.3346,
            b=0.003225,
            c=-1.9342,
            d=-4.045,
            yshift=5
        ),
        Constant(235)
    ]
    ainf_j = [
        Constant(1),
        PowerSigmoidal(
            a=1,
            b=1,
            c=0,
            d=-10,
            power=1/2
        ),
        PowerSigmoidal(
            a=1,
            b=1,
            c=-10,
            d=5,
            power=1/3
        ),
        PowerSigmoidal(
            a=1,
            b=1,
            c=-10,
            d=20,
            power=1/4
        )
    ]
    binf_j = [
        Constant(1),
        Sigmoidal(
            a=1,
            b=1,
            c=0,
            d=6
        ),
        Sigmoidal(
            a=1,
            b=1,
            c=-10,
            d=-5,
            xshift=20
        ),
        Sigmoidal(
            a=1,
            b=1,
            c=0,
            d=-20,
            xshift=60
        )
    ]
    Iapp = Constant(0)
    def v(self,j): #mV
        '''
        Nernst potentials
        '''
        return self.v_j[j - 1]
    def g(self,j): #mS/cm^2
        '''
        Conductance constants
        '''
        return self.g_j[j - 1]
    def atau(self,j,v): #ms
        '''
        Rate functions for A-side
        '''
        return self.atau_j[j - 1].run(v)
    def btau(self,j,v): #ms
        '''
        Rate functions for B-side
        '''
        return self.btau_j[j - 1].run(v)
    def ainf(self,j,v):
        '''
        Steady-state functions for A-side
        '''
        #print("ainf",j)
        return self.ainf_j[j - 1].run(v)
    def binf(self,j,v):
        '''
        Steady-state functions for B-side
        '''
        return self.binf_j[j - 1].run(v)
    def I(self,t): #applied current
        return self.Iapp.run(t)

# --- Solving the ODE and Plotting the Potential ---

def connor_stevens(t, x, p):
    v, a2, a3, a4, b2, b3, b4 = x
    def a(j):
        match j:
            case 1:
                return 1
            case 2:
                return a2
            case 3:
                return a3
            case 4:
                return a4

    def b(j):
        match j:
            case 1:
                return 1
            case 2:
                return b2
            case 3:
                return b3
            case 4:
                return b4
    summed_terms = 0
    for j in [1,2,3,4]:
        summed_terms += (p.g(j)) * (a(j) ** j) * b(j) * (v - p.v(j)) * p.disabled[j]
    dvdt = (1 / p.cm) * (p.I(t) - summed_terms)
    dadt = [0,0] # so that the indexes line up
    dbdt = [0,0]
    for j in [2,3,4]:
        dajdt = (1 / p.atau(j,v)) * (p.ainf(j,v) - a(j)) * p.disabled[j]
        dbjdt = (1 / p.btau(j,v)) * (p.binf(j,v) - b(j)) * p.disabled[j]
        dadt.append(dajdt)
        dbdt.append(dbjdt)
    out = [dvdt,dadt[2],dadt[3],dadt[4],dbdt[2],dbdt[3],dbdt[4]]
    #print(dadt,a(2),a(3),a(4),sep='\n')
    return out

def pretty_names(index):
    match index:
        case 0:
            return 'V(t)'
        case 1:
            return 'a2: K On'
        case 2:
            return 'a3: Na On'
        case 3:
            return 'a4: Combined On'
        case 4:
            return 'b2: K Off'
        case 5:
            return 'b3: Na Off'
        case 6:
            return 'b4: Combined Off'

from model_and_parameters import *

# 2. Solve Numerically
def basic_system_data():
    params = Parameters()
    V0 = [-40,0.9,0.9,0.9,0.9,0.9,0.9]
    t_span = [0, 1200]
    stepmul = 100
    t_eval = np.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0])*stepmul)
    sol = solve_ivp(connor_stevens, t_span, V0, args=(params,), dense_output=True, t_eval=t_eval, method='RK45')
    
    # 3. Plot the Results
    display_time = 150
    
    gs = gridspec.GridSpec(2,3)
    fig = plt.figure(figsize=(24,12))
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[0,1])
    ax4 = fig.add_subplot(gs[1,1])
    ax5 = fig.add_subplot(gs[0,2])
    #axs[0,0].plot(sol.t, sol.y[0])

    ax1.plot(sol.t[:display_time], sol.y[0][:display_time], label=pretty_names(0))
    ax1.set_title('Membrane Potential Time Course')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane voltage (mV)')
    ax1.grid(True)
    ax1.legend()

    for i in [1,2,3,4,5,6]:
        ax2.plot(sol.t[:display_time], sol.y[i][:display_time], label=pretty_names(i))
    ax2.set_title('Channel Behaviour Time Course')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Activated Channels Proportion')
    ax2.grid(True)
    ax2.legend()
    
    for i in [2,3,4]:
        vrange = np.linspace(-100,25,100)
        ax3.plot(vrange, params.ainf(i,vrange), label=pretty_names(i - 1))
        ax3.plot(vrange,params.binf(i,vrange), label=pretty_names(i + 2))
    ax3.set_title('Channel Behaviour vs Voltage')
    ax3.set_xlabel('Voltage (mV)')
    ax3.set_ylabel('Activated Channels Proportion')
    ax3.grid(True)
    ax3.legend()
    
    for i in [2,3,4]:
        vrange = np.linspace(-100,25,100)
        ataus = np.array([params.atau(i,v) for v in vrange])
        btaus = np.array([params.btau(i,v) for v in vrange])
        ax4.plot(vrange,ataus, label=pretty_names(i - 1))
        ax4.plot(vrange,btaus, label=pretty_names(i + 2))
    ax4.set_title('Channel Behaviour vs Voltage')
    ax4.set_xlabel('Voltage (mV)')
    ax4.set_ylabel('Rate constant (ms)')
    ax4.grid(True)
    ax4.legend()
    
    i_s = []
    for t in sol.t:
        i_s.append(params.I(t))
    
    ax5.plot(sol.t,np.array(i_s),label="Current")
    ax5.set_title("Applied Current")
    ax5.set_xlabel("Time (ms)")
    ax5.set_ylabel("Current (mA)")
    ax5.grid(True)
    ax5.legend()
    #plt.savefig("plots")
    plt.show()
    return sol

plot_steady_min_confidence = 0.7

def blob_sort(points,tolerance,verbose=False,find_confidence=False):
    bins = []
    confidences = []
    for steady in points:
        if verbose: print(steady)
        will_break = False
        for i in range(len(steady)):
            if i != 0 and (steady[i] > 1 or steady[i] < 0):
                will_break = True
            if i == 0 and (steady[0] > 500 or steady[0] < -500):
                will_break = True
        if will_break:
            break
        #print(steady)
        if len(bins) == 0:
            confidences.append(1)
            bins.append(np.array([steady]))
        else:
            is_done = False
            for b in range(len(bins)):
                norm = np.linalg.norm(bins[b][0] - steady)
                if norm < tolerance:
                    np.append(bins[b],steady)
                    confidences[b] += 1
                    is_done = True
                    break
            if not is_done:
                confidences.append(1)
                bins.append(np.array([steady]))
    confidences = np.array(confidences)
    if confidences.size != 0:
        confidences = confidences / np.max(confidences) #this line changed to reduce bias towards one solution
    if find_confidence:
        return (bins,confidences)
    else :
        return bins

def generate_points(param):
    random_start = []
    for i in [0,1,2,3,4,5,6]:
        if i == 0 :
            random_start.append((random.random() - 0.5) * 200)
        else:
            random_start.append(random.random())
    steady = fsolve(lambda x: connor_stevens(0,x,param),np.array(random_start))
    return steady

def find_steady_states(verbose=False,param=Parameters(),tolerance_max=400):
    all_bins = []
    start_time = time.time()
    if verbose: print("Stage 1")
    for tol in range(10,tolerance_max):
        tolerance = tol / 1 #to test fractional tolerances between 10, 400 by 0.1
        if verbose: print(tolerance,"time",time.time() - start_time)
        points = np.array([generate_points(param) for i in range(100)])
        bins = blob_sort(points,tolerance)
        #print(tolerance,"gives",bins)
        all_bins.append(bins)
        if verbose: print(bins)

    if verbose: print("Stage 2",time.time() - start_time)
    bins_first = []
    for b in all_bins:
        if len(b) == 1:
            if verbose: print("valid",b)
            bins_first.append(b[0])
        elif len(b) == 0:
            if verbose: print("empty")
        else:
            if verbose: print("strange",len(b),b)
    final_tolerance = 10
    if verbose: print("Stage 3",time.time() - start_time)
    for i in range(len(bins_first)):
        bins_first[i] = bins_first[i][0]
    if verbose: print(bins_first)
    final_steady_blobs,final_confidence = blob_sort(bins_first,final_tolerance,verbose=verbose,find_confidence=True)
    if verbose: print("Complete, time",time.time() - start_time)
    return (final_steady_blobs,final_confidence)

def phase_planes(sol):
    
    steady_state,confidence = find_steady_states(verbose=False,tolerance_max=110)
    print("Steady state:")
    np.set_printoptions(precision=2)
    for i in range(len(steady_state)):
        print(f" {i+1} : \033[32;1;4m{steady_state[i]} \033[0m , \033[31;1m conf {confidence[i]:.2f} \033[0m")
    
    pairs = {'2d':[[0,1],[0,2],[0,3],[0,4],[0,5],[0,6]],
            '3d':[[1,2,3],[4,5,6],[0,1,4],[0,2,5],[0,3,6]]}
    
    gs = gridspec.GridSpec(2,3)
    fig = plt.figure(figsize=(24,12))
    axes = []
    for j in [0,1]:
        for k in [0,1,2]:
            axes.append(fig.add_subplot(gs[j,k]))
    
    count = 0
    for i in pairs['2d']:
        ax1 = axes[count]
        ax1.plot(sol.y[i[0]],sol.y[i[1]])
        for s in range(len(steady_state)):
            if confidence[s] < plot_steady_min_confidence:
                print("ignoring steady state", steady_state[s],f"because confidence {confidence[s]:.2f} is less than {plot_steady_min_confidence:.2f}")
                break
            ax1.plot(steady_state[s][0][i[0]],steady_state[s][0][i[1]],'go', ms=10, label=f'Steady State, confidence {confidence[s]:.2f}')
        ax1.set_title('Phase space: ' + pretty_names(i[0]) + ' and ' + pretty_names(i[1]))
        ax1.set_xlabel(pretty_names(i[0]))
        ax1.set_ylabel(pretty_names(i[1]))
        ax1.grid(True)
        ax1.legend()
        count += 1
    
    gs2 = gridspec.GridSpec(2,3)
    fig2 = plt.figure(figsize=(24,12))
    axes2 = []
    for j in [0,1]:
        for k in [2,1,0]:
            axes2.append(fig2.add_subplot(gs2[j,k],projection='3d'))
    
    count = 0
    for i in pairs['3d']:
        ax2 = axes2[count]
        ax2.plot(sol.y[i[0]],sol.y[i[1]],sol.y[i[2]])
        for s in range(len(steady_state)):
            if confidence[s] < plot_steady_min_confidence:
                print("ignoring steady state", steady_state[s],f"because confidence {confidence[s]:.2f} is less than {plot_steady_min_confidence:.2f}")
                break
            ax2.plot(steady_state[s][0][i[0]],steady_state[s][0][i[1]],steady_state[s][0][i[2]],'go', ms=10, label=f'Steady State, confidence {confidence[s]:.2f}')
        ax2.set_title('Phase space: ' + pretty_names(i[0]) + ' and ' + pretty_names(i[1]) + ' and ' + pretty_names(i[2]))
        ax2.set_xlabel(pretty_names(i[0]))
        ax2.set_ylabel(pretty_names(i[1]))
        ax2.set_zlabel(pretty_names(i[2]))
        ax2.grid(True)
        ax2.legend()
        count += 1
    plt.show()

class Bifurcator:
    mod = None # modifier, converts parameter type to a new one with required value
    base = Parameters()
    model = connor_stevens
    p_range = np.linspace(0,100,100)
    threaded = False
    task_name = None
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
            evals_list = self.generate_modified(lambda new,i: _eigenvalue_dance_task(new,i,var_range[i]))
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
        with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
            results = [executor.submit(self._generate_modified_task,c) for c in range(len(self.p_range))]
            for r in concurrent.futures.as_completed(results):
                out_list.append(r.result())
        return out_list
    
    def _generate_modified_task(self,counter):
        i = self.p_range[counter]
        #if verbose: print("count",counter)
        new_entry = copy.deepcopy(self.base)
        new_entry = self.mod(new_entry,i)
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

def plot_bifurcation(bifurcate_results):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot()
    for b in bifurcate_results:
        for a in range(len(b[0])):
            #if b[1][a] >= 0.7:
            ax.plot(b[2],b[0][a][0][0],"o",color=(b[1][a],0,0))
    ax.set_ylabel(pretty_names(0))
    ax.set_xlabel("$V_4$")
    ax.grid(True)
    plt.show()
    
class MissingStateError(Exception):
     def __init__(self, message="This is a custom error."):
        self.message = message
        super().__init__(self.message)


def eigenvalue_plot(test,ssr,sort_var_index,min_confidence_plot=0.7,continuous_fake=False):
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
        fig = plt.figure(figsize=(18,18))
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
        labels = ["Eigenvalue real","Eigenvalue imaginary","$b_{\infty,4} x-shift$"]
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
    out.v_j[1] = m
    return out

def modifier2(p,m):
    out = p
    out.v_j[2] = m
    return out

def modifier3(p,m):
    out = p
    out.v_j[3] = m
    return out

def main():
    # basic_system_data()
    #soln = basic_system_data()
    #phase_planes(soln)
    #test = Bifurcator(modifier,Parameters(),np.linspace(-35,35,50))
    #steady_state_results = perform_bifurcation(test)
    #eigenvalue_plot(test,steady_state_results,continuous_fake=True)

    #Threaded system speed test
    #This does NOT work for eigenvalue dance as it's not really needed
    param_range = np.linspace(-80,80,100)
    thread_start = time.time()
    test1 = Bifurcator(modifier,Parameters(),param_range,True) #this True/False controls if the threading is on
    steady_state_results1 = perform_bifurcation(test1,tolerance_max=100)
    threaded_time = time.time() - thread_start
    print("1: Threaded took:",threaded_time)
    
    thread_start = time.time()
    test2 = Bifurcator(modifier2,Parameters(),param_range,True) #this True/False controls if the threading is on
    steady_state_results2 = perform_bifurcation(test2,tolerance_max=100)
    threaded_time = time.time() - thread_start
    print("2: Threaded took:",threaded_time)
    
    thread_start = time.time()
    test3 = Bifurcator(modifier3,Parameters(),param_range,True) #this True/False controls if the threading is on
    steady_state_results3 = perform_bifurcation(test3,tolerance_max=100)
    threaded_time = time.time() - thread_start
    print("3: Threaded took:",threaded_time)
    plot_bifurcation(steady_state_results1)
    plot_bifurcation(steady_state_results2)
    plot_bifurcation(steady_state_results3)
    # MARK: Finish this bit
    test1.eigenvalue_dance()
    #un_start = time.time()
    #test = Bifurcator(modifier,Parameters(),np.linspace(-80,80,50),False)
    #steady_state_results = perform_bifurcation(test,tolerance_max=50)
    #plot_bifurcation(steady_state_results)
    #unthreaded_time = time.time() - un_start
    #print("Unthreaded took:",unthreaded_time)
    #eigenvalue_plot(test,steady_state_results,0,continuous_fake=True)

    #This currently functions, but fails to graph it.
    #bifurcate_2d_timescales(2,np.linspace(0,1,100),np.linspace(0,1,100),np.linspace(0,1,100),1,Parameters())
    #timescale_phase_plots()

if __name__ == "__main__":
    main()
