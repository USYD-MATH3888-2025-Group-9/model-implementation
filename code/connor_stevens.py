# Dependencies: numpy, matplotlib, scipy

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings # Import the warnings module

# --- Setup

def linear(x,m,b):
    return m * x + b
    
def sigmoid(x,a,b,c,d):
    return a / (b + np.exp((c - x) / d))

# --- Physiological Parameters ---
class Parameters:
    cm = 14 # trial, 14pF
    disabled = [0,1,1,1,1] #blank,blank,2,3,4
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
                return -45
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
                m = (300-60) / (-50)
                b = -m * (-40) + 300
                out = linear(v,m,b)
                if v < -40:
                    out = 300
                if v > 10:
                    out = 60
                #print(out)
                return out
            case 3:
                a = 0.6335
                b = 0.05426
                c = -1.932
                d = -4.650
                out = sigmoid(v,a,b,c,d)
                #print(out)
                return out + 1 # random constant from paper
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
                return 50
            case 3:
                a = 0.3346
                b = 0.003225
                c = -1.9342
                d = -4.045
                return sigmoid(v,a,b,c,d) + 5 # random constant from paper
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
                out = np.sqrt(sigmoid(v,1,1,0,-10))
                #print(out)
                return out
            case 3:
                out = np.power(sigmoid(v,1,1,-10,5),(1 / 3)) 
                #a = 1 should be 2 according to the paper but that doesn't make sense
                #print(out)
                return out
            case 4:
                out = np.power(sigmoid(v,1,1,-10,20),(1 / 4)) 
                #a = 1 should be 2 according to the paper but that doesn't make sense
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
                #a = 1 should be 2 according to the paper but that doesn't make sense
                return sigmoid(v + 20,1,1,-10,-5)
            case 4:
                return sigmoid(v + 60,1,1,0,-20)
    @staticmethod
    def I(t): #applied current
        #return 0
        if t > 80:
            return 15
        elif t > 50:
            return -5
        else:
            return 0

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
    
# 2. Solve Numerically
params = Parameters()
V0 = [-40,0.9,0.9,0.9,0.9,0.9,0.9]
t_span = [0, 100]
t_eval = np.linspace(t_span[0], t_span[1], 300)
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


#####################
# TODO
# - Albert to search for more sensible parameters
#       - Checking from Tims dataset & trying to line things up from there for accuracy
#       - Parameter regression via least squares????
# - Kyle to start on phaseplane/box analysis
# - Tim + Calvin continuing to search for datasets
#####################

gs = gridspec.GridSpec(2,3)
fig = plt.figure(figsize=(24,12))
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[0,1])
ax4 = fig.add_subplot(gs[1,1])
ax5 = fig.add_subplot(gs[0,2])
ax6 = fig.add_subplot(gs[1,2])
#axs[0,0].plot(sol.t, sol.y[0])

ax1.plot(sol.t, sol.y[0], label=pretty_names(0))
ax1.set_title('Membrane Potential Time Course')
ax1.set_xlabel('Time (ms)')
ax1.set_ylabel('Membrane voltage (mV)')
ax1.grid(True)
ax1.legend()

for i in [1,2,3,4,5,6]:
    ax2.plot(sol.t, sol.y[i], label=pretty_names(i))
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

#plt.savefig("plots")
plt.show()
