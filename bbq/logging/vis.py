# --- Utils --- #

from tkinter import font
from bbq.utils import load_config
import yaml
import json
import pickle
import os
import numpy as np
from matplotlib import ticker

from matplotlib import pyplot as plt

def load_json(fname):
    with open(fname) as json_file:
        data = json.load(json_file)
    return data   
from pathlib import Path


def get_config_files(folder):
    file_list = []
    for file in os.listdir(folder):
        if file.endswith(".yaml"):
            file_list += [os.path.join(folder, file)]
    return file_list[::-1]

def plot_stats(data, p):
    with plt.style.context(['science','retro','notebook','y_grid']):
        fig, ax = plt.subplots(ncols=3, figsize=(15,3))
        eval_per_iter = sum([e['n_batch'] for e in p['emitters']])

        for i, stat in enumerate(data.keys()):
            x = np.array(data[stat]['itrs'])*eval_per_iter
            ax[i].plot(x, data[stat]['vals'])
            ax[i].set_title(stat)     
            ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))

        arch_size = max_bins(p)
        ax[0].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=arch_size, decimals=0))
        ax[0].set_ylim([0,arch_size*1.1])
        ax[0].yaxis.set_major_locator(ticker.FixedLocator(np.linspace(0.0,arch_size,5)))

        plt.setp(ax.flat, xlabel='Iterations')
    return fig, ax

def plot_pulse(pulse, p):
    stat = [norm_pulse(np.array(p)) for p in pulse]
    with plt.style.context(['science','retro','notebook','y_grid']):
        fig, ax = plt.subplots(ncols=len(p['emitters']),figsize=(15,3))

    event_label = ['Not Added', 'Improved', 'Discovered']
    for i, pulse in enumerate(stat):
        eval_per_iter = sum([e['n_batch'] for e in p['emitters']])
        x = np.arange(len(pulse))[::10]*eval_per_iter
        y = pulse[::10,:]
        for j in range(3):
            y[:,j] = moving_average(y[:,j]) 
        ax[i].stackplot(x, y[:,0], y[:,1], y[:,2], labels=event_label)
        ax[i].set_title(p['emitters'][i]['name'], fontsize=16)
        ax[i].set_ylim([0.8,1.0])
        ax[i].ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax[i].yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
        ax[i].yaxis.set_major_locator(ticker.FixedLocator(np.linspace(0.8,1.0,5)))

        if i > 0:
            ax[i].set_yticklabels([])
        plt.setp(ax.flat, xlabel='Iterations')            

    ax[0].set_ylabel('Proportion of\nCreated Solutions')      
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, event_label,loc='center', bbox_to_anchor=(0.5, -0.2), 
              fancybox=True, shadow=True, ncol=3, fontsize=16)
    return fig, ax
  
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