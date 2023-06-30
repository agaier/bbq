import os
import numpy as np
import json
from matplotlib.ticker import FormatStrFormatter
from matplotlib import pyplot as plt
from pathlib import Path


def load_json(fname):
    with open(fname) as json_file:
        data = json.load(json_file)
    return data 

def get_stats(exp, stat_name):
    stat = {}
    for e in exp:
        e = Path(e)
        stat_list = []    
        stat[e.name] = {}

        dir_name = str(e)
        rep_folders = next(os.walk(dir_name))[1]    
        for rep in rep_folders:
            fname = e/rep/'metrics.json'
            #print(fname)
            stat_tmp = load_json(fname)
            stat_list.append(stat_tmp[stat_name]['vals'])
        stat[e.name]['full'] = np.stack(stat_list)

    # Get Medians and Quartiles        
    for e in exp:  
        e = Path(e)  
        stat[e.name]['median'] = np.median(stat[e.name]['full'], axis=0)
        stat[e.name]['25th'] = np.quantile(stat[e.name]['full'], q=.25, axis=0)
        stat[e.name]['75th'] = np.quantile(stat[e.name]['full'], q=.75, axis=0)    
        
    return stat
    

def plot_comparison(stat, obj_labels=None):
    algs = list(stat.keys())
    sample_stat = stat[algs[0]]['median']

    if len(sample_stat.shape) > 1:
        n_metric = stat[algs[0]]['median'].shape[1]
    else:
        n_metric = 1
        
    if n_metric > 1:
        ax = plot_multi_comparison(stat, obj_labels)
    else:
        ax = plot_single_comparison(stat)        

    return ax


def plot_multi_comparison(stat, obj_labels=None):
    algs = list(stat.keys())
    n_alg = len(algs)
    n_metric = stat[algs[0]]['median'].shape[1]
    
    if obj_labels is None:
        obj_labels = [f"Objective {i}" for i in np.arange(n_metric)]

    # Differentiate lines and metrics
    line_style = ['-', ':', '-.', '--']
    C = [plt.cm.tab10c(c) for c in np.linspace(0.0,.4,n_metric)][::-1] # Colors

    # Set axis labels
    i=0
    fig, host = plt.subplots(figsize=(8,4),dpi=100)
    pars = [host.twinx() for _ in range(n_metric)] # Create additional y scales for each metric
    host.set_xlabel("Iterations")

    # Objectives
    spine_pos = np.arange(n_metric)*66
    for i, par in enumerate(pars):
        par.set_ylabel(obj_labels[i])
        for j, alg in enumerate(algs):
            y0 = stat[alg]['median']
            y1 = stat[alg]['25th']
            y2 = stat[alg]['75th']
            x = np.arange(len(y0))+1
            par.fill_between(x, y1[:,i], y2[:,i], color=C[i], linestyle=line_style[j], alpha=0.2)
            p, = par.plot(x, y0[:,i], color=C[i], label=obj_labels[i], linewidth=2, linestyle=line_style[j])        
        par.spines['right'].set_position(('outward', spine_pos[i]))
        par.spines['right'].set_color(C[i])
        par.spines['right'].set_linewidth(3)
        par.yaxis.label.set_color(C[i])   
        par.yaxis.set_major_formatter(FormatStrFormatter('%2.e'))

    host.axes.get_yaxis().set_ticks([])
    # Dummy lines for Legend
    alg_lines = []
    for i, alg in enumerate(algs):    
        p, = host.plot(np.nan, np.nan, color=C[1], label=algs[i], linestyle=line_style[i], linewidth=2)
        alg_lines.append(p)    
    host.legend(handles=alg_lines, loc='upper center', ncol=len(alg_lines), fancybox=True, bbox_to_anchor=(0.5, -0.15), shadow=True)  
    return host

def plot_single_comparison(stat):
    algs = list(stat.keys())
    n_alg = len(algs)

    # Differentiate algorithms
    C = [plt.cm.Paired(c) for c in np.linspace(0.0,.3,n_alg)][::-1] # Colors

    # Plot
    fig, ax = plt.subplots(figsize=(8,4),dpi=100)
    for i, alg in enumerate(algs):
        y0 = stat[alg]['median']
        y1 = stat[alg]['25th']
        y2 = stat[alg]['75th']
        x = np.arange(len(y0))+1
        ax.fill_between(x, y1, y2, color=C[i], alpha=0.3)
        p, = ax.plot(x, y0, color=C[i], linewidth=2)     
    ax.yaxis.tick_right()

    # Legend    
    alg_lines = []
    for i, alg in enumerate(algs):    
        p, = ax.plot(np.nan, np.nan, color=C[i], label=algs[i], linewidth=2)
        alg_lines.append(p)    
    ax.legend(handles=alg_lines, loc='upper center', ncol=len(alg_lines), fancybox=True, bbox_to_anchor=(0.5, -0.15), shadow=True)  
    return ax