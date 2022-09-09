# --- Utils --- #

import json
import os
import numpy as np

from bbq.utils import load_config
from matplotlib import cm, ticker
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable


# File I/O
def load_json(fname):
    with open(fname) as json_file:
        data = json.load(json_file)
    return data   

def get_config_files(folder):
    file_list = []
    for file in os.listdir(folder):
        if file.endswith(".yaml"):
            file_list += [os.path.join(folder, file)]
    return file_list[::-1]

# High level plotting functions
def plot_stats(data, p):
    with plt.style.context(['bbq_line']):
        fig, ax = plt.subplots(ncols=3, figsize=(15,3))
        eval_per_iter = sum([e['batch_size'] for e in p['emitters']])
        
        for i, stat in enumerate(list(data.keys())[:3]):
            x = np.array(data[stat]['itrs'])*eval_per_iter
            ax[i].plot(x, data[stat]['vals'])
            ax[i].set_title(stat)     
            ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax[i].legend(data[stat]['label'])
            if np.max(data[stat]['vals']) > 1000:
                ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))

        arch_size = max_bins(p)
        ax[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=arch_size, decimals=0))
        ax[0].set_ylim([0,arch_size*1.1])
        ax[0].yaxis.set_major_locator(ticker.FixedLocator(np.linspace(0.0,arch_size,5)))

        plt.setp(ax.flat, xlabel='Evaluations')
    return fig, ax

def plot_pulse(pulse, p):
    stat = [norm_pulse(np.array(p)) for p in pulse]
    with plt.style.context(['bbq_line']):
        fig, ax = plt.subplots(ncols=len(p['emitters']))

    event_label = ['Not Added', 'Improved', 'Discovered']
    for i, pulse in enumerate(stat):
        eval_per_iter = sum([e['batch_size'] for e in p['emitters']])
        if pulse.shape[0] > 100:
            x = np.arange(len(pulse))[::10]*eval_per_iter*10
            y = pulse[::10,:]
        else:
            x = np.arange(len(pulse))*eval_per_iter
            y = pulse            
        for j in range(3):
            y[:,j] = moving_average(y[:,j]) 

        ax[i].stackplot(x, y[:,0], y[:,1], y[:,2], labels=event_label)
        ax[i].set_title(p['emitters'][i]['name'], fontsize=16)
        ax[i].set_ylim([0.75,1.0])
        ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax[i].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        ax[i].yaxis.set_major_locator(ticker.FixedLocator(np.linspace(0.75,1.0,6)))

        if i > 0:
            ax[i].set_yticklabels([])
        plt.setp(ax.flat, xlabel='Iterations')            

    ax[0].set_ylabel('Proportion of\nCreated Solutions')      
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, event_label,loc='center', bbox_to_anchor=(0.5, -0.2), 
              fancybox=True, shadow=True, ncol=3, fontsize=16)
    return fig, ax

def plot_map(Z, p, ax=None, bin_ticks=False, n_colors=None):
    if ax is None:
        fig,ax = plt.subplots(figsize=(4,4),dpi=150)    
    with plt.style.context('me_grid'):        
        if n_colors is None:
            im = ax.imshow(Z, cmap='YlGnBu')
        else:
            cmap = cm.get_cmap('YlGnBu', n_colors)
            im = ax.imshow(Z, cmap=cmap)
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        for t in cb.ax.get_yticklabels():
            t.set_horizontalalignment('right')   
            t.set_x(3.0)

        # Grid
        set_map_grid(ax, Z, bin_ticks=bin_ticks, **p)        
        ax.set(xlabel = p['desc_labels'][0], ylabel= p['desc_labels'][1])
                
    return ax

# Multiple Quantities
def plot_ys(x, ys, labels, ax=None, reverse_y=False):
    if ax is None:
        fig, host = plt.subplots(figsize=(8,5),dpi=150)
    else:
        host = ax
    ys= np.rollaxis(ys,1) # [quantities X series]
    C = [plt.cm.Set2(c) for c in np.linspace(0,1,len(ys))][::-1] # Colors
    pars = [host.twinx() for _ in range(len(ys[1:]))] # Create additional y scales

    # Set axis labels
    i=0
    host.set_xlabel("Iterations")

    p, = host.plot(x, ys[i], color=C[i], label=labels[i], linewidth=2, linestyle=':')
    lns = [p]
    host.set_ylabel('Archive Size')
    host.spines['left'].set_color(C[i])
    host.spines['left'].set_linewidth(3)
    host.yaxis.label.set_color(C[i])   

    for i, par in enumerate(pars,1):
        par.set_ylabel(labels[i])
        p, = par.plot(x, ys[i], color=C[i], label=labels[i], linewidth=2, linestyle='--')
        lns.append(p)
        par.spines['right'].set_position(('outward', i*70))
        par.spines['right'].set_color(C[i])
        par.spines['right'].set_linewidth(3)
        par.yaxis.label.set_color(C[i])   
        if reverse_y:
            par.invert_yaxis()
    host.legend(handles=lns, loc='upper center', ncol=len(lns), fancybox=True, 
                bbox_to_anchor=(0.5, -0.125), shadow=True)   
    return host


# Plotting utilties
# Ticks
def set_map_grid(ax, Z, bin_ticks=False, desc_bounds=0, grid_res=0, **_):
    map_x, map_y = Z.shape[0], Z.shape[1]
    xticks = -0.5+np.arange(map_x)
    yticks = -0.5+np.arange(map_y)  

    xlabels = np.linspace(desc_bounds[0][0],
                        desc_bounds[0][1],
                        grid_res[0])

    ylabels = np.linspace(desc_bounds[1][0],
                        desc_bounds[1][1],
                        grid_res[1])
    #xlabels = np.round(xlabels,1).astype(int)
    #ylabels = np.round(ylabels,0).astype(int)    

    abbrev_xlabels = ['']*map_x
    abbrev_ylabels = ['']*map_y


    if bin_ticks:
        skip = 3
        abbrev_xlabels[::skip] = np.arange(0, map_x, skip)
        abbrev_ylabels[::skip] = np.arange(0, map_y, skip)
    else:
        abbrev_xlabels[1], abbrev_xlabels[-1] = xlabels[0], xlabels[-1]
        abbrev_ylabels[1], abbrev_ylabels[-1] = ylabels[0], ylabels[-1]


    x_formatter, x_locator = FixedFormatter(abbrev_xlabels), FixedLocator(xticks)
    y_formatter, y_locator = FixedFormatter(abbrev_ylabels), FixedLocator(yticks)

    ax.xaxis.set_major_locator(x_locator)
    ax.xaxis.set_major_formatter(x_formatter)

    ax.yaxis.set_major_locator(y_locator)
    ax.yaxis.set_major_formatter(y_formatter)

    grid_thick = 15/np.max(Z.shape)
    ax.grid(linewidth=grid_thick)
    ax.tick_params(direction="out", width=grid_thick, length=grid_thick*2)
    return ax


# Numerical utilities
def norm_pulse(pulse):
    n_children  = np.sum(pulse,axis=1)
    norm_factor = np.tile(n_children,(3,1)).T
    pulse /= norm_factor
    return pulse

def moving_average(y, window_width = 5):
    cumsum_vec = np.cumsum(np.insert(y, 0, 0)) 
    ma_vec = (cumsum_vec[window_width:] - cumsum_vec[:-window_width]) / window_width
    y[:len(ma_vec)] = ma_vec
    return y

def max_bins(p):
    archive = p['archive']
    if archive['type'] == "Grid":
        return np.prod(archive['grid_res'])
    if archive['type'] == "CVT":
        return archive['n_bins']
    return ValueError("Invalid Archive Type -- can't calculate max bins")




# -- ---------------------------------------------------------------------- -- #
# # TODO: Make these into nice jupyerlab usable functions
# # Image
# def map_to_image(Z, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(4,4), dpi=150)        
#     img = ax.imshow(np.rollaxis(Z,1).astype(float), cmap='YlGnBu')
#     cbar = plt.colorbar(img,ax=ax)
#     ax.invert_yaxis()
#     return ax

# def view_map(Z, p, ax=None, bin_ticks=False):
#     if ax is None:
#         fig,ax = plt.subplots(figsize=(4,4),dpi=150)    
#     with plt.style.context('me_grid'):        
#         im = ax.imshow(Z, cmap='YlGnBu')

#         # Colorbar
#         divider = make_axes_locatable(ax)
#         cax = divider.append_axes("right", size="5%", pad=0.1)
#         cb = plt.colorbar(im, cax=cax)
#         for t in cb.ax.get_yticklabels():
#             t.set_horizontalalignment('right')   
#             t.set_x(3.0)

#         # Grid
#         set_map_grid(ax, Z, bin_ticks=bin_ticks, **p)
        
#         ax.set(xlabel = p['desc_labels'][0], ylabel= p['desc_labels'][1])
                
#     return ax

