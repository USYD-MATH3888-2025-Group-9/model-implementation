#!/usr/bin/python3


from scipy.integrate import solve_ivp
import matplotlib.gridspec as gridspec
import warnings
from matplotlib.collections import LineCollection

from model_and_parameters import * 
import machine_config

import wfdb
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


dataentry = "a1t18"
PATH = f"../../datasets/physionet.org/files/sgamp/1.0.0/raw/{dataentry}"

def gen_records(PATH):
    try:
        RECORD = wfdb.rdrecord(PATH)
        df = pd.DataFrame(RECORD.p_signal, columns=RECORD.sig_name)
        # Division by 10 to bring down to correct values (see documentation )
        df["Vmembrane"] = df["Vmembrane"].div(10)
        print(df)
    except FileNotFoundError:
        print(f"{PATH} DOES NOT EXIST!")
    except Exception as E:
        print(f"An error has occured: {E}")
    return df

def main():
    df = gen_records(PATH=PATH)

    
    # Cleaning up values
    V_VALS = df["Vmembrane"][1:]
    I_VALS = df["Istim"][1:]
    numvals = len(V_VALS)
    max_t = (numvals) / 125000
    print(f"Final t : {max_t}")
    T_VALS = np.linspace(0,numvals,numvals)


    # Fetching our model
    params = Parameters()
    V0 = [-40,0.5,0.5,0.5,0.5,0.5,0.5]
    t_span = [0, 20000]
    t_eval = np.linspace(t_span[0], t_span[1], numvals)
    soln = solve_ivp(connor_stevens, t_span, V0, args=(params,), dense_output=True, t_eval=t_eval, method='RK45')

    # Plotting setup
    gs = gridspec.GridSpec(2,2,height_ratios=[4,1])
    fig1 = plt.figure(figsize=(16,10))
    ax1_1 = fig1.add_subplot(gs[0,0])
    ax2_1 = fig1.add_subplot(gs[1,0])
    ax3_1 = fig1.add_subplot(gs[0,1])
    ax4_1 = fig1.add_subplot(gs[1,1])

    tspace = np.linspace(0,2,numvals)
    scaled_soln_vals = soln.y[0] / 1000 # scale so that both sets of values are in the same range
    # Plotting Results
    ax1_1.plot(tspace,scaled_soln_vals, color="orange", label="Model Results")
    ax1_1.set_title("Membrane potential from model")
    
    ax2_1.plot(tspace, [Parameters.I(Parameters, i) for i in tspace])
    ax2_1.set_title("Stimulating current from model")
    
    ax3_1.plot(tspace, V_VALS, label=f"Data from squid ({dataentry})")
    ax3_1.set_title("Membrane potential from data")

    ax4_1.plot(tspace, I_VALS, label=f"Applied current from data ({dataentry})")
    ax4_1.set_title("Stimulating current from data")

    fig1.suptitle(f"Data and our model : {machine_config.author}")

    #fig1.legend(loc='center', bbox_to_anchor=(0.9, 0.925), bbox_transform=fig.transFigure)
    
    gs = gridspec.GridSpec(2,1,height_ratios=[4,1])
    fig2 = plt.figure(figsize=(16,10))
    ax1_2 = fig2.add_subplot(gs[0,0])
    ax2_2 = fig2.add_subplot(gs[1,0])

    ax1_2.plot(tspace,scaled_soln_vals, label = "Model values")
    ax1_2.plot(tspace,V_VALS, label = "Data values")
    ax1_2.legend(loc="center", bbox_to_anchor=(0.05,0.95))
    ax2_2.set_title("Log difference (error)")
    ax2_2.plot(tspace, np.log(np.abs(scaled_soln_vals - V_VALS)), label = "log difference")
    
    fig2.suptitle(f"Comparison of membrane potential from data\n and our model : {machine_config.author}")
    
    
    plt.show()
    
if __name__ == "__main__":
    main()
