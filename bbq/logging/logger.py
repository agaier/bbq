from logging import raiseExceptions
from ribs.visualize import grid_archive_heatmap

import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil
import numpy as np
from bbq.logging.plotting import  map_to_image, set_map_grid, plot_ys
from humanfriendly import format_timespan




class RibsLogger():
    def __init__(self, p, save_meta=False, copy_config=True, clear=True, rep=None):
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
        if rep is None:
            self.log_dir = Path(f'log/{p["task_name"]}/{p["exp_name"]}')
        else:
            self.log_dir = Path(f'log/{p["task_name"]}/{p["exp_name"]}/{rep}')
        if clear:
            if self.log_dir.exists() and self.log_dir.is_dir():
                shutil.rmtree(self.log_dir)
        self.log_dir.mkdir(parents=True,exist_ok=True)

        if copy_config:
            self.copy_config()
        

    def log_metrics(self, archive, itr, time, save_all=False):
        ''' Calls all logging and visualization functions '''
        self.update_metrics(archive, itr)
        if (itr%self.p['print_rate'] == 0) or self.p['print_rate'] == 1:
            n_evals = self.p['n_batch']*self.p['n_emitters']
            self.print_metrics(archive, itr, n_evals, time)  
            with (self.log_dir / f"metrics.json").open("w") as file:
                json.dump(self.metrics, file, indent=2)        

        if (itr%self.p['plot_rate']==0) or save_all:
            self.plot_metrics()
            self.plot_obj(archive)

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

    def print_metrics(self, archive, itr, eval_per_iter, time):
        ''' Print metrics to command line '''    
        # TODO: add improvement as % of batch
        qd = np.array(self.metrics['QD Score']['vals'][-1])
        print(f"Iter: {itr}" \
            +f" | Eval: {itr*eval_per_iter}" \
            +f" | Size: {archive.stats.num_elites}" \
            +f" | QD: {np.round(qd)}" \
            +f" | Time/Itr: {format_timespan(time)}")         

    def plot_metrics(self):
        ''' Line plot of archive size and QD Score ''' 
        _, ax = plt.subplots(figsize=(8,5),dpi=150)
        x = np.array(self.metrics['QD Score']['itrs'])
        ys = np.c_[np.array(self.metrics['Archive Size']['vals'][1:]),
                    np.array(self.metrics['QD Score']['vals'])]     
        labels = ["Archive Size"]+["QD Score"]     
        ax = plot_ys(x, ys, labels, ax=ax) # <-- the actual plotting
        fname = str(self.log_dir / "LINE_Metrics.png")
        plt.savefig(fname,bbox_inches='tight')
        plt.clf(); plt.close()

    def save_archive(self, archive, outdir, f_type='numpy', export_meta=True):  
        ''' Saves entire archive as a pandas file, optionally w/metadata '''             
        if f_type == 'numpy':
            out_archive = archive.as_numpy(include_metadata=export_meta)
            out_archive = np.rollaxis(out_archive,-1)
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


    def plot_obj(self, archive):
        labels = ['Fitness']
        A = np.rollaxis(archive.as_numpy(),-1)
        val = A[0]
        fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=150)
        ax = map_to_image(val, ax=ax)
        ax = set_map_grid(ax, val, **self.p) 
        ax.set(xlabel = self.p['desc_labels'][0], 
               ylabel= self.p['desc_labels'][1])
        plt.subplots_adjust(hspace=0.4)
        fig.savefig(str(self.log_dir / f"MAP_Fitness.png"))
        plt.clf(); plt.close()

    def copy_config(self):
        config_paths = [self.p['base_config_path'], self.p['exp_config_path']]
        for config_file in config_paths:
            src = Path(config_file)
            dest = self.log_dir/src.name
            shutil.copy(str(src), str(dest)) # for python <3.8

    def zip_results(self):
        file_name = Path(str(self.log_dir)+"_result")
        shutil.make_archive(file_name, 'zip', self.log_dir)

