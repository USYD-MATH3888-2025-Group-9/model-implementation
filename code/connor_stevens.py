# Dependencies: numpy, matplotlib, scipy

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import warnings # Import the warnings module

# --- Setup

def linear(x,m,b):
    return m * x + b
    
def sigmoid(x,a,b,c,d):
    return a / (b + np.exp((c - x) / d))

# --- Physiological Parameters ---
class Parameters:
    cm = 14 # trial, 14pF
    @staticmethod
    def v(j): #mV
        '''
        Nernst potentials
        '''
        match j:
            case 1:
                return -40
            case 2:
                return -60
            case 3:
                return 45
            case 4:
                return -63
    @staticmethod
    def g(j): #mS/cm^2
        '''
        Conductance constants
        '''
        match j:
            case 1:
                return 0.049
            case 2:
                return 10 
            case 3:
                return 21 
            case 4:
                return 10
    @staticmethod
    def atau(j,v): #ms
        '''
        Rate functions for A-side
        '''
        #print("atau",j)
        match j:
            case 1:
                return 1 #unused
            case 2:
                #capped linear
                m = (0.3-0.06) / (-50)
                b = -m * (-40) + 0.3
                out = linear(v,m,b)
                if v < -40:
                    out = 0.3
                if v > -10:
                    out = 0.06
                #print(out)
                return out
            case 3:
                a = 0.6335
                b = 0.05426
                c = -1.932
                d = -4.650
                out = sigmoid(v,a,b,c,d)
                #print(out)
                return out
            case 4:
                return 12 
    @staticmethod
    def btau(j,v): #ms
        '''
        Rate functions for B-side
        '''
        match j:
            case 1:
                return 1 #unused
            case 2:
                return 0.05
            case 3:
                a = 0.3346
                b = 0.003225
                c = -1.9342
                d = -4.045
                return sigmoid(v,a,b,c,d)
            case 4:
                return 235 
    @staticmethod
    def ainf(j,v):
        '''
        Steady-state functions for A-side
        '''
        #print("ainf",j)
        match j:
            case 1:
                return 1
            case 2:
                out = np.sqrt(sigmoid(v,1,1,0,6))
                #print(out)
                return out
            case 3:
                out = np.power(sigmoid(v,2,1,-10,20),(1 / 3))
                #print(out)
                return out
            case 4:
                out = np.power(sigmoid(v,2,1,-10,20),(1 / 4))
                #print(out)
                return out
    @staticmethod
    def binf(j,v):
        '''
        Steady-state functions for B-side
        '''
        match j:
            case 1:
                return 1
            case 2:
                return sigmoid(v,1,1,0,6)
            case 3:
                return sigmoid(v,2,1,-10,20)
            case 4:
                return sigmoid(v - 60,1,1,0,6)
    @staticmethod
    def I(t):
        return 1

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
        summed_terms += (p.g(j)) * (a(j) ** j) * b(j) * (v - p.v(j))
    dvdt = (1 / p.cm) * (p.I(t) - summed_terms)
    dadt = [0,0] # so that the indexes line up
    dbdt = [0,0]
    for j in [2,3,4]:
        dajdt = (1 / p.atau(j,v)) * (p.ainf(j,v) - a(j))
        dbjdt = (1 / p.btau(j,v)) * (p.binf(j,v) - b(j))
        dadt.append(dajdt)
        dbdt.append(dbjdt)
    out = [dvdt,dadt[2],dadt[3],dadt[4],dbdt[2],dbdt[3],dbdt[4]]
    #print(dadt,a(2),a(3),a(4),sep='\n')
    return out
    
# 2. Solve Numerically
params = Parameters()
V0 = [-20,0.1,0.1,0.1,0.9,0.9,0.9]
t_span = [0, 50]
t_eval = np.linspace(t_span[0], t_span[1], 500)
sol = solve_ivp(connor_stevens, t_span, V0, args=(params,), dense_output=True, t_eval=t_eval, method='RK45')

# 3. Plot the Results

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

plt.figure(figsize=(12, 6))
plt.plot(sol.t, sol.y[0], label=pretty_names(0))
plt.title('Membrane Potential Time Course')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane voltage (mV)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for i in [1,2,3,4,5,6]:
    plt.plot(sol.t, sol.y[i], label=pretty_names(i))
plt.title('Channel Behaviour Time Course')
plt.xlabel('Time (ms)')
plt.ylabel('Activated Channels Proportion')
plt.grid(True)
plt.legend()
plt.show()
