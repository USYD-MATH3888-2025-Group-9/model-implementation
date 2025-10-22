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

# Variables specific to machines for things like number of threads etc
import machine_config
# Model + params, pulled out for neatness &/or portability
from model_and_parameters import *
# Basic version of the plotting stuff
from plotting_general import *
# -
from bifurcator import *
# - 
from timescales import *


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

def basic_plots():
    soln = basic_system_data()
    phase_planes(soln)
    # test = Bifurcator(modifier,Parameters(),np.linspace(-35,35,50))
    # steady_state_results = perform_bifurcation(test)
    # eigenvalue_plot(test,steady_state_results,continuous_fake=True)


def bifurcation_plots():
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

def main():
    time_series_plot_final_report()
    phase_plane_plots_final_report()


if __name__ == "__main__":
    main()
