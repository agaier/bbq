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
def plot_stats(data, p, vertical=False, ax=None):
    with plt.style.context(['bbq_line']):
        if ax is None:
            if vertical:
                fig, ax = plt.subplots(nrows=3, figsize=(5,15))
                plt.subplots_adjust(hspace=0.3)
            else:
                fig, ax = plt.subplots(ncols=3, figsize=(15,5))           
            
        for i, stat in enumerate(list(data.keys())[:3]):
            x = np.array(data[stat]['itrs']) # * eval_per_iter
            ax[i].plot(data[stat]['itrs'], data[stat]['vals'])
            ax[i].set_title(stat, fontsize=14)     
            ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax[i].legend(data[stat]['label'])
            if np.max(data[stat]['vals']) > 1000:
                ax[i].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        arch_size = max_bins(p)
        ax[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=arch_size, decimals=0))
        ax[0].set_ylim([0,arch_size*1.1])
        ax[0].yaxis.set_major_locator(ticker.FixedLocator(np.linspace(0.0,arch_size,5)))


        plt.setp(ax.flat, xlabel='Evaluations')
        for i in range(len(ax)):
            ax[i].xaxis.label.set_size(14)

def plot_pulse(pulse, p):
    stat = [norm_pulse(np.array(p)) for p in pulse]
    with plt.style.context(['bbq_line']):
        fig, ax = plt.subplots(ncols=len(p['emitters']))
        if type(ax) is not np.ndarray:
            ax = np.array([ax])

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

from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter


def plot_map(Z, p, ax=None, bin_ticks=False, n_colors=16, cmap='cividis'):
    if ax is None:
        fig,ax = plt.subplots(figsize=(4,4),dpi=150)    
    with plt.style.context('bbq_gridmap'):      
        cmap = cm.get_cmap(cmap, n_colors)
        im = ax.imshow(Z, cmap=cmap, origin='lower')
        # -- Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        if n_colors < 12:
            n_cticks = n_colors+1
        else:
            n_cticks = 5
        fmt = lambda x, pos: '{:1.2}'.format(x)
        c_ticks = np.linspace(np.nanmin(Z), np.nanmax(Z), n_cticks)
        cb = plt.colorbar(im, cax=cax, ticks=c_ticks, format=FuncFormatter(fmt))
        cb.ax.locator_params(nbins=n_colors)
        for t in cb.ax.get_yticklabels():
            t.set_horizontalalignment('right')   
            t.set_x(3.0)
        # -- rid
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
    C = [plt.cm.Set2(c) for c in np.linspace(0,1,len(ys))] # Colors
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

def view_map(Z, p, ax=None, bin_ticks=False):
    if ax is None:
        fig,ax = plt.subplots(figsize=(4,4),dpi=150)    
    with plt.style.context('bbq_gridmap'):        
        im = ax.imshow(Z, cmap='YlGnBu')

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
        # Set title to fitness_labels if exists
        if 'fitness_labels' in p.keys():
            ax.set_title(p['fitness_labels'][0])
        else:
            ax.set_title("Fitness")
                
    return ax

def view_by_bin(coord, meta, visualize, ax, inset_coord=[1.5,0,1,1]):
    solution = meta[coord[::-1]][0]
    axins = ax.inset_axes(inset_coord)
    visualize(solution, ax=axins)
    ax.indicate_inset([coord[0]-0.5, coord[1]-0.5, 1, 1],inset_ax=axins, 
                        lw=2,edgecolor='r',facecolor='r', alpha=0.3, fill=False, zorder=5)

def view_best(fit, meta, visualize, ax):
    best_idx = np.unravel_index(np.nanargmax(fit), fit.shape)[::-1]
    view_by_bin(best_idx, meta, visualize, ax)

def view_solutions(fit, meta, visualize, p, plot_per_side=3, bin_list=None):
    # Set inset plot positions and source solutions
    x_pts, y_pts = (plot_per_side, plot_per_side)
    
    if plot_per_side == 4:
        inset_size = 0.8
        x_pos = 0.25+np.linspace(-1.5, 1.5, x_pts)
        y_pos = 0.25+np.linspace(-1.5, 1.5, y_pts)
        xx,yy = np.meshgrid(x_pos, y_pos)
        plot_pos = np.c_[xx.flatten(),yy.flatten(), inset_size*np.ones(xx.flatten().shape), inset_size*np.ones(xx.flatten().shape)]
        plot_pos*= 0.8

    else:
        inset_size = 1
        x_pos = np.linspace(-1.5, 1.5, x_pts)*0.8
        y_pos = np.linspace(-1.5, 1.5, y_pts)*0.8
        xx,yy = np.meshgrid(x_pos, y_pos)
        plot_pos = np.c_[xx.flatten(),yy.flatten(), inset_size*np.ones(xx.flatten().shape), inset_size*np.ones(xx.flatten().shape)]


    # Default bins
    if bin_list is None:
        x_res, y_res = fit.shape
        x_coord = np.round(np.linspace(0,x_res,x_pts+2)).astype(int)[1:-1]
        y_coord = np.round(np.linspace(0,y_res,y_pts+2)).astype(int)[1:-1]

        xx,yy = np.meshgrid(x_coord, y_coord)
        nx2 = np.c_[xx.flatten(),yy.flatten()]
        bin_list = list(zip(nx2[:,0], nx2[:,1]))
        bin_list = [(b[0]+np.random.randint(4)-2,b[1]+np.random.randint(4)-2) for b in bin_list] # add some jitter

    fig,ax = plt.subplots(figsize=(9,4),dpi=100)    
    plot_map(fit, p, bin_ticks=True, ax=ax)
    n_plots = x_pts*y_pts
    for i in range(n_plots):
        if (n_plots == 9 and i == 4) or (n_plots == 16 and np.any(i == np.array([5,6,9,10]))):
            bin_list[i] = np.nan
            pass    
        else:        
            view_by_bin(bin_list[i], meta, visualize, ax, inset_coord=plot_pos[i])

    return bin_list

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