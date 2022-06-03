import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil
import numpy as np
from bbq.logging.plotting import  map_to_image, set_map_grid, plot_ys
from humanfriendly import format_timespan
import pickle
from bbq.logging import RibsLogger

class MCLogger(RibsLogger):
    def __init__(self, p, save_meta=False, copy_config=True, clear=True, rep=0, zip=False):
        super().__init__(p, save_meta, copy_config, clear, rep, zip)

    
    def update_metrics(self, archive, itr):
        ''' Adds current iterations metrics to running record '''        
        self.metrics["Archive Size"]["itrs"].append(itr)
        self.metrics["Archive Size"]["vals"].append(sum(archive.num_elites))
        self.metrics["QD Score"]["itrs"].append(itr)
        self.metrics["QD Score"]["vals"].append(sum(archive.qd_score)) 

    def print_metrics(self, archive, itr, eval_per_iter, time):
        ''' Print metrics to command line '''    
        # TODO: add improvement as % of batch
        qd = np.array(self.metrics['QD Score']['vals'][-1])
        print(f"Iter: {itr}" \
            +f" | Eval: {itr*eval_per_iter}" \
            +f" | Size: {archive.num_elites}" \
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

    def plot_obj(self, archives):
        for i, archive in enumerate(archives.archives):
            labels = ['Fitness']
            A = np.rollaxis(archive.as_numpy(),-1)
            val = A[0]
            fig,ax = plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=150)
            ax = map_to_image(val, ax=ax, clim=(150,200))
            ax = set_map_grid(ax, val, **archive.p) 
            ax.set(xlabel = archive.p['desc_labels'][0], 
                   ylabel = archive.p['desc_labels'][1])
            plt.subplots_adjust(hspace=0.4)
            fig.savefig(str(self.log_dir / f"{i}-MAP_Fitness.png"))
            plt.clf(); plt.close()        

    def save_archive(self, archive, itr='', f_type='numpy', export_meta=True):  
        ''' Saves entire archive as a pandas file, optionally w/metadata '''             
        outdir = self.archive_dir
        if f_type == 'numpy':
            out_archives = archive.as_numpy(include_metadata=export_meta)
            for i, out_archive in enumerate(out_archives):
                out_archive = np.rollaxis(out_archive,-1)
                if export_meta:
                    np.save(outdir / f'{i}-archive_{itr}.npy', out_archive[0])
                    np.save(outdir / f'{i}-archive_meta_{itr}.npy', out_archive[1])
                else:
                    np.save(outdir / f'{i}-archive_{itr}.npy', out_archive)
        elif f_type == 'pandas':
            raise NotImplementedError
            final_archive = archive.as_pandas(include_metadata=True)
            final_archive.to_pickle(outdir / f'archive_{itr}.pd')
        else:
            raise ValueError("Invalid file type for archive (numpy/pandas)")     
