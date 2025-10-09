#!/usr/bin/python3

import model_and_parameters as cs

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridpsec
from scipy.integrate import solve_ivp
from scipy.differentiate import jacobian


t_span = [0,1000]
numsteps = (t_span[1] - t_span[0])* 1000
t_eval = np.linspace(t_span[0], t_span[1], numsteps)

def handle():
    V0 = [-40,0.9,0.9,0.9,0.9,0.9,0.9]
    params = cs.Parameters()
    soln = solve_ivp(cs.connor_stevens, t_span, V0, args=(params,), dense_output=True, t_eval=t_eval, method='RK45')

    fig1, ax1 = plt.subplots()
    i, j = 0,6

    ax1.plot(soln.y[i], soln.y[j])
    ax1.set_xlabel(cs.pretty_names(i))
    ax1.set_ylabel(cs.pretty_names(j))
    if j != 0:
        ax1.set_ylim(0,1)


'''
Timescales are 
T1; Slow        :       a2, b4
T2; Medium      :       b2, b3
T3; Fast        :       a3, a4, V



V = (I_app + C~) / ( C )
C       = (g2 a2^2 b2 + g3 a3^3 b3 + g4 a4^4 b2)
C~      = (g2 a2^2 b2 v2 + g3 a3^3 b3 v3 + g4 a4^4 b4 v4)
    the parameters that C, C~ take will vary depending on the timescale and are noted below

--- T1 -------------------------------------


--- T2 -------------------------------------
a3 -> a3inf(V)
a4 -> a4inf(V)

C   : (b2,b3)   -> RR^2
C~  : (b2,b3)   -> RR^2

--- T3 -------------------------------------
a3 -> a3inf(V)
a4 -> a4inf(V)
b2 -> b2inf(V)
b3 -> b3inf(V)

C   : (a3, b4)  -> S
C~  : (a3, b4)  -> S

'''





def main():
    handle()
    plt.show()

if __name__ == "__main__":
    main()