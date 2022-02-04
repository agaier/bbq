from logging import raiseExceptions
from ribs.visualize import grid_archive_heatmap

import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil
import numpy as np

class RibsLogger():
    def __init__(self, p, save_meta=False, copy_config=None, clear=True):
        self.p = p
        self.save_meta = save_meta        
        self.metrics = {
            "Archive Size": {
                "itrs": [0],
                "vals": [0],  # Starts at 0.
            },
            "Mean Fitness": {
                "itrs": [],
                "vals": [],  # Does not start at 0.
            },      
            "QD Score": {
                "itrs": [],
                "vals": [],  # Does not start at 0.
            },              
        }       
        # Reset log folder
        self.log_dir = Path(f'log/{p["name"]}'); 
        if clear:
            if self.log_dir.exists() and self.log_dir.is_dir():
                shutil.rmtree(self.log_dir)
        self.log_dir.mkdir(parents=True,exist_ok=True)

        if copy_config is not None:
            self.copy_config(copy_config)
        

    def log_metrics(self, archive, itr, save_all=False):
        ''' Calls all logging and visualization functions '''
        self.update_metrics(archive, itr)
        if (itr%self.p['print_rate'] == 0) or self.p['print_rate'] == 1:
            self.print_metrics(archive, itr, self.p['n_batch']*self.p['n_emitters'])
            with (self.log_dir / f"metrics.json").open("w") as file:
                json.dump(self.metrics, file, indent=2)        

        if (itr%self.p['plot_rate']==0) or save_all:
            self.plot_metrics()
            self.plot_map(archive, self.log_dir, self.p['desc_labels'])

        if (itr%self.p['save_rate']==0) or save_all:
            self.save_archive(archive, self.log_dir, export_meta=self.save_meta)

    def update_metrics(self, archive, itr):
        ''' Adds current iterations metrics to running record '''        
        self.metrics["Archive Size"]["itrs"].append(itr)
        self.metrics["Archive Size"]["vals"].append(archive.stats.num_elites)
        self.metrics["Mean Fitness"]["itrs"].append(itr)
        self.metrics["Mean Fitness"]["vals"].append(archive.stats.obj_mean)
        self.metrics["QD Score"]["itrs"].append(itr)
        self.metrics["QD Score"]["vals"].append(archive.stats.qd_score)   

    def print_metrics(self, archive, itr, eval_per_iter):
        ''' Print metrics to command line '''    
        print(f"Iter: {itr}" \
            +f" | Eval: {itr*eval_per_iter}" \
            +f" | Size: {archive.stats.num_elites}" \
            +f" | Mean: {archive.stats.obj_mean:.2f}" \
            +f" | Max:  {archive.stats.obj_max:.2f}"\
            +f" | QD:   {archive.stats.qd_score:.2f}")        

    def plot_metrics(self):
        ''' Creates a plot for each matrix to visualize during and after run '''    
        for metric in self.metrics:
            plt.plot(self.metrics[metric]["itrs"], self.metrics[metric]["vals"])
            plt.title(metric)
            plt.xlabel("Iteration")
            plt.savefig(
                str(self.log_dir / f"{metric.lower().replace(' ', '_')}.png"))
            plt.clf()  

    def plot_map(self, archive, log_dir, desc_labels=None):
        ''' Creates a heatmap of the current archive '''        
        fig,ax = plt.subplots(figsize=(8,6),dpi=150)
        grid_archive_heatmap(archive, ax=ax)
        if desc_labels is not None:
            ax.set(xlabel= desc_labels[0][0],
                   ylabel= desc_labels[1][0])
        fig.savefig(str(log_dir / f"MAP.png"))   
        plt.clf()  
        plt.close()
        
    def save_archive(self, archive, outdir, f_type='numpy', export_meta=True):  
        ''' Saves entire archive as a pandas file, optionally w/metadata '''             
        if f_type == 'numpy':
            out_archive = archive.as_numpy(include_metadata=export_meta)
            if export_meta:
                np.save(outdir / f'archive.npy', out_archive[0])
                np.save(outdir / f'archive_meta.npy', out_archive[1])
            else:
                np.save(outdir / f'archive.npy', out_archive)
        elif f_type == 'pandas':
            final_archive = archive.as_pandas(include_metadata=True)
            final_archive.to_pickle(outdir / f'archive.pd')
        else:
            raise ValueError("Invalid file type for archive (numpy/pandas)")     

    def copy_config(self, config_file):
        src = Path(config_file)
        dest = self.log_dir/src.name
        shutil.copy(str(src), str(dest)) # for python <3.8

    def zip_results(self):
        file_name = Path(str(self.log_dir)+"_result")
        shutil.make_archive(file_name, 'zip', self.log_dir)



