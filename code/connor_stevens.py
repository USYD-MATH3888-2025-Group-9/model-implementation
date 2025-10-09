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

import model_and_parameters as cs


# 2. Solve Numerically
def basic_system_data():
    params = cs.Parameters()
    V0 = [-40,0.9,0.9,0.9,0.9,0.9,0.9]
    t_span = [0, 1200]
    stepmul = 100
    t_eval = np.linspace(t_span[0], t_span[1], (t_span[1] - t_span[0])*stepmul)
    sol = solve_ivp(cs.connor_stevens, t_span, V0, args=(params,), dense_output=True, t_eval=t_eval, method='RK45')
    
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

    ax1.plot(sol.t[:display_time], sol.y[0][:display_time], label=cs.pretty_names(0))
    ax1.set_title('Membrane Potential Time Course')
    ax1.set_xlabel('Time (ms)')
    ax1.set_ylabel('Membrane voltage (mV)')
    ax1.grid(True)
    ax1.legend()

    for i in [1,2,3,4,5,6]:
        ax2.plot(sol.t[:display_time], sol.y[i][:display_time], label=cs.pretty_names(i))
    ax2.set_title('Channel Behaviour Time Course')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Activated Channels Proportion')
    ax2.grid(True)
    ax2.legend()
    
    for i in [2,3,4]:
        vrange = np.linspace(-100,25,100)
        ax3.plot(vrange, params.ainf(i,vrange), label=cs.pretty_names(i - 1))
        ax3.plot(vrange,params.binf(i,vrange), label=cs.pretty_names(i + 2))
    ax3.set_title('Channel Behaviour vs Voltage')
    ax3.set_xlabel('Voltage (mV)')
    ax3.set_ylabel('Activated Channels Proportion')
    ax3.grid(True)
    ax3.legend()
    
    for i in [2,3,4]:
        vrange = np.linspace(-100,25,100)
        ataus = np.array([params.atau(i,v) for v in vrange])
        btaus = np.array([params.btau(i,v) for v in vrange])
        ax4.plot(vrange,ataus, label=cs.pretty_names(i - 1))
        ax4.plot(vrange,btaus, label=cs.pretty_names(i + 2))
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
    steady = fsolve(lambda x: cs.connor_stevens(0,x,param),np.array(random_start))
    return steady

def find_steady_states(verbose=False,param=cs.Parameters(),tolerance_max=400):
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
        ax1.set_title('Phase space: ' + cs.pretty_names(i[0]) + ' and ' + cs.pretty_names(i[1]))
        ax1.set_xlabel(cs.pretty_names(i[0]))
        ax1.set_ylabel(cs.pretty_names(i[1]))
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
        ax2.set_title('Phase space: ' + cs.pretty_names(i[0]) + ' and ' + cs.pretty_names(i[1]) + ' and ' + cs.pretty_names(i[2]))
        ax2.set_xlabel(cs.pretty_names(i[0]))
        ax2.set_ylabel(cs.pretty_names(i[1]))
        ax2.set_zlabel(cs.pretty_names(i[2]))
        ax2.grid(True)
        ax2.legend()
        count += 1
    plt.show()

def unscrambled(current,past,past2):
    '''
    Sorts the current eigenvalues based on past experience.
    '''
    #grind through the options
    min_distance = None
    min_order = None
    counter = 0
    past_grads = []
    for j in range(len(past)):
        past_grads.append((past[j].imag - past2[j].imag) / (past[j].real - past2[j].real))
    for option in itertools.combinations(range(len(past)),len(current)):
        this_distance = 0
        this_option = option
        c_count = 0
        for j in option:
            past_current_grad = (past[j].imag - current[j].imag) / (past[j].real - current[j].real)
            this_distance += abs(past_grads[j] - past_current_grad)
            c_count += 1
        if min_distance == None or min_distance >= this_distance:
            min_distance = this_distance
            min_order = option
        
    #using the minimum, figure out the order.
    print(min_order)
    out = []
    for i in min_order:
        out.append(current[i])
    return out
    
class Bifurcator:
    mod = None # modifier, converts parameter type to a new one with required value
    base = cs.Parameters()
    model = cs.connor_stevens
    p_range = np.linspace(0,100,100)
    def steady_states(self,verbose=False,max_tolerance=50):
        '''
        Compute steady states for this bifurcation setup
        '''
        def _steady_states_task(new,i,verbose):
            steady,confidence = find_steady_states(param=new,tolerance_max=max_tolerance)
            confidence_sorted = confidence.tolist()
            confidence_sorted.sort(reverse=True)
            confidence_finder = {}
            for j in range(len(steady)):
                confidence_finder[steady[j][0][0]] = confidence[j]
            steady.sort(reverse=True,key=lambda x: confidence_finder[x[0][0]])
            if verbose: print(steady,confidence_sorted)
            return [steady,confidence_sorted,i]
        return self.generate_modified(lambda new,i: _steady_states_task(new,i,verbose),verbose)
    def eigenvalue_dance(self,var_range,verbose=False):
        '''
        Find and trace the eigenvalues of the Jacobian for the value
        of the variables in var_range.
        '''
        def _eigenvalue_dance_task(new,i,var_at_new,verbose):
            jacob = jacobian(lambda x: cs.connor_stevens(0,x,new),var_at_new).df
            eig_data = eig(jacob,right=True)
            evectors = eig_data[1]
            evals = eig_data[0]
            if verbose: print(f"eigenvalues \033[31;1m {evals} \033[m ")
            return [np.array(evals),i]
        evals_list = self.generate_modified(lambda new,i: _eigenvalue_dance_task(new,i,var_range[i],verbose),verbose)
        #now fix the ordering
        second = False
        last_evals = np.array([])
        last2_evals = np.array([])
        final_evals = []
        for ev in evals_list:
            e = ev[0]
            print(e,last_evals,last2_evals)
            fixed_e = unscrambled(e,last_evals,last2_evals) if len(last_evals) != 0 and len(last2_evals) != 0 else e
            last2_evals = last_evals
            last_evals = fixed_e
            final_evals.append([fixed_e,ev[1]])
        return final_evals

    def generate_modified(self,task,verbose):
        '''
        Computes and returns the results of an arbitrary task
        over the given parameter set.

        task: (Parameters,p) -> Any for p in p_range
        '''
        out_list = []
        counter = 0
        for i in self.p_range:
            if verbose: print(f"count\033[95m {counter} \033[m")
            new_entry = copy.deepcopy(self.base)
            new_entry = self.mod(new_entry,i)
            if verbose: print(f"running {self.p_range[counter]} ")
            out_list.append(task(new_entry,i))
            counter += 1
        return out_list
    
    def __init__(self,mod,base,p_range):
        self.mod = mod
        self.base = base
        self.p_range = p_range

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

def perform_bifurcation(test):
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot()
    bifurcate_results = test.steady_states(verbose=True)
    for b in bifurcate_results:
        for a in range(len(b[0])):
            ax.plot(b[2],b[0][a][0][0],"go",color=(b[1][a],0,0))
    ax.set_ylabel(cs.pretty_names(0))
    ax.set_xlabel("$V_4$")
    ax.grid(True)
    plt.show()
    return bifurcate_results

class MissingStateError(Exception):
     def __init__(self, message="This is a custom error."):
        self.message = message
        super().__init__(self.message) # Call the parent Exception's constructor


def eigenvalue_plot(test,ssr,continuous_fake=False):
    steady_states = {}
    last_index = None
    for i in ssr:
        if len(i[0]) != 0:
            last_index = i[2]
            steady_states[i[2]] = i[0][0][0]
        else:
            if continuous_fake:
                steady_states[i[2]] = steady_states[last_index]
                print("Fake value used for",i[2],steady_states[i[2]])
            else:
                raise MissingStateError(message="Missing state info for value " + str(i))
    gs = gridspec.GridSpec(2,2)
    fig = plt.figure(figsize=(18,18))
    ax = fig.add_subplot(gs[0,0],projection="3d")
    ax2 = fig.add_subplot(gs[1,0])
    ax3 = fig.add_subplot(gs[0,1])
    ax4 = fig.add_subplot(gs[1,1])
    bifurcate_results2 = test.eigenvalue_dance(steady_states)
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
    labels = ["Eigenvalue real","Eigenvalue imaginary","$b_{\infty,3} x-shift$",]
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

def modifier(p,m):
    out = p
    out.binf_j[1] = cs.Sigmoidal(a=1,b=1,c=-10,d=-5,xshift=m)
    return out

###########################
# Actually run the system #
###########################

def main():
    # basic_system_data()
    soln = basic_system_data()
    phase_planes(soln)
    test = Bifurcator(modifier,cs.Parameters(),np.linspace(-35,35,50))
    steady_state_results = perform_bifurcation(test)
    eigenvalue_plot(test,steady_state_results,continuous_fake=True)


if __name__ == "__main__":
    main()

#it still swaps the values occasionally
#damnit
