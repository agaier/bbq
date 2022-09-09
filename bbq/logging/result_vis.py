from os import listdir
from os.path import isfile, join
import json
import numpy as np
from matplotlib import pyplot as plt

def load_json(fname):
    with open(fname) as json_file:
        data = json.load(json_file)
    return data   

def collect_stats(folder, stat_key):
    # Load all stats into single np_array
    rep_folders = [f for f in listdir(folder) if not isfile(join(folder, f))]
    val_list = []
    for rep in rep_folders:
        fname = f"{folder}{rep}/metrics.json"
        tmp = load_json(fname)
        if stat_key == "Mean Fitness":
            val_list.append(np.array(tmp['Fitness']['vals'])[:,0])
            itr_array = np.array(tmp['Fitness']['itrs'])
        elif stat_key == "Max Fitness":
            val_list.append(np.array(tmp['Fitness']['vals'])[:,1])            
            itr_array = np.array(tmp['Fitness']['itrs'])
        else:
            val_list.append(tmp[stat_key]['vals'])   
            itr_array = np.array(tmp[stat_key]['itrs'])

    val_array = np.stack(val_list)
    return val_array, itr_array

def summarize(stat):
    summary = {}
    summary['median'] = np.median(stat, axis=0)
    summary['25th'] = np.quantile(stat, q=.25, axis=0)
    summary['75th'] = np.quantile(stat, q=.75, axis=0)
    return summary

def get_rep_stats(folder):
    stat_names = ['QD Score', 'Archive Size', 'Mean Fitness', 'Max Fitness']
    stat_dict = {}
    for stat in stat_names:
        val_array, itr_array = collect_stats(folder, stat)
        stat_dict[stat] = summarize(val_array)
        stat_dict[stat]['itr'] = itr_array
        stat_dict[stat]['name'] = stat
    return stat_dict

def plot_rep(stat, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,3))     
    if label is None:
        l_median = ax.plot(stat['itr'], stat['median'], label=stat['name'])
    else:
        l_median = ax.plot(stat['itr'], stat['median'], label=label)
    l_shade  = ax.fill_between(stat['itr'], stat['25th'], stat['75th'], color=l_median[0].get_color(), alpha=0.2)
    ax.set_title(stat['name'], fontsize=14)
    ax.set_xlabel('Evaluations', fontsize=12)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    if np.max(stat['median']) > 1000:
        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0)) 
    ax.legend()        

def compile_result(data_path):
    exp_folders = [f for f in listdir(data_path) if not isfile(join(data_path, f))]
    result_dict = {}
    for folder in exp_folders:
        result_dict[folder] = get_rep_stats(f"{data_path}/{folder}/") 
    return result_dict

def plot_result(result_dict, stat_name, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,3))  
    for alg in result_dict.keys():
        stats = result_dict[alg]
        plot_rep(stats[stat_name], ax=ax, label=alg)
