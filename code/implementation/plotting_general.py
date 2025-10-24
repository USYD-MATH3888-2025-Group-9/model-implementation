from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


from model_and_parameters import * 
import machine_config

def basic_system_data():
    params = Parameters()
    V0 = [-40,0.5,0.5,0.5,0.5,0.5,0.5]
    t_span = [0, 3000]
    numsteps = (t_span[1] - t_span[0])#*stepmul
    t_eval = np.linspace(t_span[0], t_span[1], numsteps)
    sol = solve_ivp(connor_stevens, t_span, V0, args=(params,), dense_output=True, t_eval=t_eval, method='RK45')
    
    # 3. Plot the Results
    display_time = 3000
    
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
    fig.suptitle(f"Basic plots - {machine_config.author}")
    plt.show()
    #plt.savefig("plots")
    return sol


def time_series_plot_final_report():
    params = Parameters()
    V0 = [-40,0.9,0.8,0.8,0.1,0.2,0.2]
    t_span = [0, 3000]
    numsteps = (t_span[1] - t_span[0])#*stepmul
    t_eval = np.linspace(t_span[0], t_span[1], numsteps)
    sol = solve_ivp(connor_stevens, t_span, V0, args=(params,), dense_output=True, t_eval=t_eval, method='RK45')
    
    displaytime = 3000

    gs = gridspec.GridSpec(2,1,height_ratios=[5,1])
    fig = plt.figure(figsize=(16,10))


    ax2 = fig.add_subplot(gs[0,0])
    ax3 = fig.add_subplot(gs[1,0])



    for i in [1,2,3,4,5,6]:
        ax2.plot(sol.t[:displaytime], sol.y[i][:displaytime], label=pretty_names(i))
    ax2.set_title('Channel Behaviour Time Course')
    ax2.set_xlabel('Time (ms)')
    ax2.set_ylabel('Activated Channels Proportion')
    ax2.grid(True)

    ax2.legend(loc='center', bbox_to_anchor=(0.925, 0.9), bbox_transform=fig.transFigure)

    ax3.plot(sol.t[:displaytime],sol.y[0][:displaytime], label=pretty_names(0))
    ax3.set_title("Membrane potential v. time")
    ax3.set_xlabel("Time (ms)")
    ax3.set_ylabel("Membrane voltage (mV)")


    fig.suptitle(f"Channel behaviour and Voltage v. time- {machine_config.author}")

    plt.show()

def phase_plane_plots_final_report():
    params = Parameters()
    V0 = [-40,0.5,0.5,0.5,0.5,0.5,0.5]
    t_span = [0, 3000]
    numsteps = (t_span[1] - t_span[0])*stepmul
    t_eval = np.linspace(t_span[0], t_span[1], numsteps)
    sol = solve_ivp(connor_stevens, t_span, V0, args=(params,), dense_output=True, t_eval=t_eval, method='RK45')
    
    displaytime = 3000

    pairs = [[0,2],[0,3],[0,5]]
    

    gs = gridspec.GridSpec(1,3,width_ratios=[1,1,1])
    fig = plt.figure(figsize=(16,10))
    

    axes = []
    for j in [0]:
        for k in [0,1,2]:
            axes.append(fig.add_subplot(gs[j,k]))
    
    
    count = 0
    for i in pairs:
        ax = axes[count]
        ax.plot(sol.y[i[0]],sol.y[i[1]])
        ax.set_title('Phase space: ' + pretty_names(i[0]) + ' and ' + pretty_names(i[1]))
        ax.set_xlabel(pretty_names(i[0]))
        ax.set_ylabel(pretty_names(i[1]))
        ax.grid(True)
        # ax.legend()
        count += 1

    fig.suptitle(f"Selected phase planes - {machine_config.author}")

    plt.show()

