import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil
import numpy as np
from bbq.logging.vis import view_map
from humanfriendly import format_timespan
import pickle
from bbq.logging.vis import plot_stats, plot_pulse, norm_pulse
from ribs.visualize import cvt_archive_heatmap


class RibsLogger():
    def __init__(self, p, save_meta=False, copy_config=True, clear=True, rep=0, zip=False, root_path=""):
        self.p = p
        self.save_meta = save_meta      
        self.zip = zip  
        self.metrics = {
            "Archive Size": {
                "itrs": [0],
                "vals": [0],  # Starts at 0.
                "label": ['Filled Bins'],
            },
            "Fitness": {
                "itrs": [],
                "vals": [],  # Does not start at 0.
                "label": ['Mean Fitness', 'Max Fitness'],
            },                  
            "QD Score": {
                "itrs": [],
                "vals": [],  # Does not start at 0.
                "label": ['QD Score'],
            },    
            "Improvement Ratio": {
                "itrs": [],
                "vals": [],  # Does not start at 0.
                "label": ['Improvement'],
            },              
        }       
        # Set log folders
        self.log_dir = Path(f'{root_path}/log/{p["task_name"]}/{p["exp_name"]}/{rep}')
        if clear:
            if self.log_dir.exists() and self.log_dir.is_dir():
                shutil.rmtree(self.log_dir)
        self.log_dir.mkdir(parents=True,exist_ok=True)

        self.archive_dir = self.log_dir/'archive'
        self.archive_dir.mkdir(parents=True,exist_ok=True)

        if copy_config:
            self.copy_config()
        
    def final_log(self, domain, archive, itr, time):
        ''' Final log method, allows for final visualization/evaluation options '''
        self.log_metrics(domain, archive, itr, time, save_all=True)
        if self.zip:
            self.zip_results()

    def log_metrics(self, opt, d, itr, time, save_all=False):
        ''' Calls all logging and visualization functions '''
        archive = opt.archive
        emitter = opt.emitters
        self.update_metrics(archive, emitter, itr)
        if (itr==1 or itr%self.p['print_rate'] == 0) or self.p['print_rate'] == 1:
            n_evals = sum([emitter['batch_size'] for emitter in self.p['emitters']])
            self.print_metrics(archive, itr, n_evals, time)  
            with (self.log_dir / f"metrics.json").open("w") as file:
                json.dump(self.metrics, file, indent=2)        

        if (itr%self.p['plot_rate']==0) or save_all:
            self.plot_metrics()
            self.plot_obj(archive)
            self.plot_pulses(emitter)

        if (itr%self.p['save_rate']==0) or save_all:
            self.save_archive(archive, itr=itr)
            self.save_pulse(emitter)

    def save_pulse(self, emitter):        
        pulses = [e.pulse[1:,:] for e in emitter] # skip 0
        if np.sum(np.stack(pulses)) == 0: return # no pulse data for emitters
        with open(self.log_dir / 'emitter_pulse.pkl', 'wb') as f:
            pickle.dump(pulses, f)         

    def plot_pulses(self, emitter):
        # - Prep Data
        pulses = [e.pulse[1:,:] for e in emitter] # skip 0th
        if np.sum(np.stack(pulses)) == 0: return # no pulse data for emitters
        fig, ax = plot_pulse(pulses, self.p)     
        fname = str(self.log_dir / "PULSE_emitter.png")
        plt.savefig(fname,bbox_inches='tight')
        plt.clf(); plt.close()

    def archive_to_numpy(self, archive):
        """Only works for grid archives"""
        if not hasattr(archive, 'boundaries'):
            return {}
        grid_res = [len(a)-1 for a in archive.boundaries]
        n_beh    = archive._behavior_dim        

        if archive.use_objects:
            genome_archive = np.full(np.r_[grid_res, 1], np.nan, dtype=object)
        else:
            genome_archive = np.full(np.r_[grid_res, self.p['n_params']], np.nan)
        fit_archive    = np.full(np.r_[grid_res], np.nan)
        desc_archive   = np.full(np.r_[grid_res, n_beh], np.nan)
        meta_archive   = np.full(np.r_[grid_res, 1], np.nan, dtype=object)

        for elite in archive:
            fit_archive   [elite.idx[0], elite.idx[1]]   = elite.obj
            desc_archive  [elite.idx[0], elite.idx[1],:] = elite.beh
            genome_archive[elite.idx[0], elite.idx[1],:] = elite.sol
            meta_archive  [elite.idx[0], elite.idx[1],:] = [elite.meta]

        archive_dict = {'fit': fit_archive,    'desc': desc_archive, 
                        'x'  : genome_archive, 'meta': meta_archive}
        return archive_dict
        
    def save_archive(self, archive, itr=''):  
        ''' Saves archive as a set of numpy files'''
        outdir = self.archive_dir
        if hasattr(archive, 'boundaries'):
            archive_dict = self.archive_to_numpy(archive)
            for key, value in archive_dict.items():
                np.save(outdir / f'{key}_{itr}.npy', value)
                np.save(outdir / f'_{key}.npy', value)
        else:
            archive_pandas = archive.as_pandas(include_metadata=True)
            archive_pandas.to_pickle(outdir / f'archive_{itr}.pd')
            archive_pandas.to_pickle(outdir / f'_archive.pd')

    def update_metrics(self, archive, emitter, itr):
        ''' Adds current iterations metrics to running record '''        
        fitness = [archive.stats.obj_mean, np.nanmax(archive._objective_values)]
        pulses = [e.pulse[1:,:] for e in emitter]
        combined_pulse = np.sum(pulses,axis=0)
        itr_evals = np.sum(combined_pulse[-1])
        imp_ratio = np.sum(combined_pulse[-1][1:])/itr_evals
        total_evals = self.metrics["Archive Size"]["itrs"][-1]+itr_evals

        self.metrics["Archive Size"]["itrs"].append(total_evals)
        self.metrics["Archive Size"]["vals"].append(archive.stats.num_elites)
        self.metrics["Fitness"]["itrs"].append(total_evals)
        self.metrics["Fitness"]["vals"].append(fitness)      
        self.metrics["QD Score"]["itrs"].append(total_evals)
        self.metrics["QD Score"]["vals"].append(archive.stats.qd_score)   
        self.metrics["Improvement Ratio"]["itrs"].append(total_evals)
        self.metrics["Improvement Ratio"]["vals"].append(imp_ratio)

    def print_metrics(self, archive, itr, eval_per_iter, time):
        ''' Print metrics to command line '''    
        qd = np.array(self.metrics['QD Score']['vals'][-1])
        imp_ratio = self.metrics["Improvement Ratio"]["vals"][-1]
        print(f"Iter: {str(itr).rjust(3, '0')}" \
            +f" | Eval: {itr*eval_per_iter}" \
            +f" | Size: {archive.stats.num_elites}" \
            +f" | QD: {qd:E}" \
            +f" | Imp Ratio: {imp_ratio:.2f}" \
            +f" | Time/Itr: {format_timespan(time)}")         

    def plot_metrics(self):
        ''' Line plot of recorded metrics ''' 
        plot_stats(self.metrics, self.p, vertical=True)
        fname = str(self.log_dir / "LINE_Metrics.png")
        plt.savefig(fname,bbox_inches='tight')
        plt.clf(); plt.close()

    def plot_obj(self, archive):
        # TODO: clean up to be universal for archive types
        archive_dict = self.archive_to_numpy(archive)
        fig,ax = plt.subplots(figsize=(4,4),dpi=150)
        if (archive_dict):
            ax = view_map(archive_dict['fit'], self.p['archive'], ax=ax)
        else: # if it is a CVT archive, use pyribs default for now:
            cvt_archive_heatmap(archive, ax=ax, cmap='YlGnBu')
        fig.savefig(str(self.log_dir / f"MAP_Fitness.png"))
        plt.clf(); plt.close()

    def copy_config(self):
        for i, config_file in enumerate(self.p['config_files']):
            src = Path(config_file)
            dest = self.log_dir/f'{i}_{src.name}'
            shutil.copy(str(src), str(dest)) # for python <3.8

    def zip_results(self):
        file_name = Path(str(self.log_dir)+"_result")
        shutil.make_archive(file_name, 'zip', self.log_dir)
