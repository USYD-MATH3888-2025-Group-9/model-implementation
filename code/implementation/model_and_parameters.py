import numpy as np

# Basic model

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


def pulse(t, t_start, val, length=0.001):
    if t > t_start and t < t_start + length:
        return val
    else:
        return 0

def v_shift(t,j):
    match j:
        case 0:
            return 0
        case 1:
            return 0
        case 2:
            return 0
        case 3:
            return 0 + pulse(t,1,2.5)



def Iapp_ext(t):
    base = 7.65
    return base + pulse(t,1,25) + pulse(t,1.8,25)
    
# --- Physiological Parameters ---
class Parameters:
    cm = 14 # trial, 14pF
    disabled = [0,1,1,1,1] #blank,blank,2,3,4
    frozen_vars = [0.1,0.1,0.1,0.1,0.1,0.1] #a2,a3,a4,b2,b3,b4
    v_j = [-40,-60,45,-63]
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
    Iapp =  Constant(0)
    def v(self,t,j): #mV
        '''
        Nernst potentials
        '''
        return self.v_j[j - 1] + v_shift(t,j-1)
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
        return self.Iapp.run(t) + Iapp_ext(t)


# Solution parameters
stepmul = 100

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
        summed_terms += (p.g(j)) * (a(j) ** j) * b(j) * (v - p.v(t,j)) * p.disabled[j]
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


# --- Prettying things up
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

# Multiple timescale stuff

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
    
