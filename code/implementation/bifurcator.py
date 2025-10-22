
import numpy as np
from scipy.optimize import fsolve
from scipy.differentiate import jacobian
from scipy.linalg import eig

import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import time
import itertools
import copy
import concurrent.futures


from model_and_parameters import *
import machine_config



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
    
    #steady_state,confidence = find_steady_states(verbose=False,tolerance_max=110)
    print("Steady state:")
    np.set_printoptions(precision=2)
    # for i in range(len(steady_state)):
    #     print(f" {i+1} : \033[32;1;4m{steady_state[i]} \033[0m , \033[31;1m conf {confidence[i]:.2f} \033[0m")
    
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
    #     for s in range(len(steady_state)):
    #         if confidence[s] < plot_steady_min_confidence:
    #             print("ignoring steady state", steady_state[s],f"because confidence {confidence[s]:.2f} is less than {plot_steady_min_confidence:.2f}")
    #             break
    #         ax1.plot(steady_state[s][0][i[0]],steady_state[s][0][i[1]],'go', ms=10, label=f'Steady State, confidence {confidence[s]:.2f}')
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
#        for s in range(len(steady_state)):
#            if confidence[s] < plot_steady_min_confidence:
#                print("ignoring steady state", steady_state[s],f"because confidence {confidence[s]:.2f} is less than {plot_steady_min_confidence:.2f}")
#                break
#            ax2.plot(steady_state[s][0][i[0]],steady_state[s][0][i[1]],steady_state[s][0][i[2]],'go', ms=10, label=f'Steady State, confidence {confidence[s]:.2f}')
        ax2.set_title('Phase space: ' + pretty_names(i[0]) + ' and ' + pretty_names(i[1]) + ' and ' + pretty_names(i[2]))
        ax2.set_xlabel(pretty_names(i[0]))
        ax2.set_ylabel(pretty_names(i[1]))
        ax2.set_zlabel(pretty_names(i[2]))
        ax2.grid(True)
        ax2.legend()
        count += 1
    
    fig.suptitle(f"2-d phase plots - {machine_config.author}")
    fig2.suptitle(f"3-d phase plots - {machine_config.author}")
    plt.show()

def unscrambled(current, past, past2):
    '''
    Sorts the current eigenvalues based on past ezperience
    '''
    min_distance = None
    min_order = None
    counter = 0
    for option in itertools.combinations(range(len(past)),len(current)):
        this_distance = 0
        this_option = option
        count = 0
        for j in option:
            this_distance += abs(past[count] - current[j])
            count += 1
        if min_distance == None or min_distance >= this_distance:
            min_distance = this_distance
            min_order = option
        out = []
        for i in min_order:
            out.append(current[i])
        print(f"{min_order} {out} {past}")
    return out

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
            return self.generate_modified(lambda new,i: Bifurcator._steady_states_task(new,i))
    
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
        with concurrent.futures.ProcessPoolExecutor(max_workers=machine_config.num_threads) as executor:
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
    ax.set_title(f"$V_$$ vs {pretty_names(0) - {machine_config.author}}")
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
        fig.suptitle(f"Eigenvalues plotted over bifurcation parameter [TODO TEST PARAMETER] - {machine_config.author}")
        ax.grid(True)
        ax2.grid(True)
        ax3.grid(True)
        ax4.grid(True)
        ax.legend()
        plt.show()


