import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd

dataset = pd.read_csv("giant_squid_hh_sim.csv")


def plot_values(dataset):
    df = pd.DataFrame(dataset)

    tvals = df[df.columns[0]]
    VmVvals = df[df.columns[1]]
    mvals = df[df.columns[2]]
    hvals = df[df.columns[3]]
    nvals = df[df.columns[4]]
    currentvals = df[df.columns[5]]

    gs = gridspec.GridSpec(2,1, height_ratios=[5,1])
    fig = plt.figure(figsize=(18,6))
    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[1,0])
    ax1.plot(tvals, mvals, label = "m")
    ax1.plot(tvals, hvals, label = "h")
    ax1.plot(tvals, nvals, label = "n")
    #ax1.plot(tvals, hvals + nvals)
    ax1.set_xlabel("time (ms)")
    ax1.set_ylabel("Proportion of channels which are open")
    ax1.grid()
    ax1.legend()

    ax2.plot(tvals, currentvals)
    ax2.set_xlabel("time (ms)")
    ax2.set_ylabel("Current (uA cm^-2)")
    ax2.set_ybound(-1,12)

    plt.savefig("squiddataplots")
    #plt.show()

def main():
    plot_values(dataset)
   

if __name__ == "__main__":
    main()