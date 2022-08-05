import matplotlib.pyplot as plt
from pathlib import Path
import json
import shutil
import numpy as np
from bbq.logging.plotting import  map_to_image, set_map_grid, plot_ys
from humanfriendly import format_timespan
import pickle



class RibsLogger():
    def __init__(self, p, save_meta=False, copy_config=True, clear=True, rep=0, zip=False):
        self.p = p
        self.save_meta = save_meta      
        self.zip = zip  
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
        # if rep is None:
        #     self.log_dir = Path(f'log/{p["task_name"]}/{p["exp_name"]}')
        # else:
        self.log_dir = Path(f'log/{p["task_name"]}/{p["exp_name"]}/{rep}')
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
        self.zip_results()

    def log_metrics(self, opt, d, itr, time, save_all=False):
        ''' Calls all logging and visualization functions '''
        archive = opt.archive
        emitter = opt.emitters
        self.update_metrics(archive, itr)
        if (itr%self.p['print_rate'] == 0) or self.p['print_rate'] == 1:
            n_evals = sum([emitter['n_batch'] for emitter in self.p['emitters']])
            self.print_metrics(archive, itr, n_evals, time)  
            with (self.log_dir / f"metrics.json").open("w") as file:
                json.dump(self.metrics, file, indent=2)        

        if (itr%self.p['plot_rate']==0) or save_all:
            self.plot_metrics()
            #self.plot_obj(archive)
            self.plot_pulse(emitter)

        if (itr%self.p['save_rate']==0) or save_all:
            self.save_archive(archive, itr=itr)
            self.save_pulse(emitter)

    def save_pulse(self, emitter):        
        pulses = [e.pulse[1:,:] for e in emitter] # skip 0
        if np.sum(np.stack(pulses)) == 0: return # no pulse data for emitters
        with open(self.log_dir / 'emitter_pulse.pkl', 'wb') as f:
            pickle.dump(pulses, f)

    def plot_pulse(self, emitter):
        # - Prep Data
        pulses = [e.pulse[1:,:] for e in emitter] # skip 0th
        if np.sum(np.stack(pulses)) == 0: return # no pulse data for emitters
        def norm_pulse(pulse):
            n_children  = np.sum(pulse,axis=1)
            norm_factor = np.tile(n_children,(3,1)).T
            pulse /= norm_factor
            return pulse
        stat = [norm_pulse(np.array(p)) for p in pulses]
        
        # - Plot Data
        rows = int(np.ceil(len(stat)/2))
        fig,ax = plt.subplots(nrows=rows,ncols=2,figsize=(8,4),dpi=100)
        ax = ax.flatten()
        event_label = ['NOT ADDED', 'IMPROVED', 'DISCOVERED']
        emitter_label = [em.name for em in emitter]
        for i, pulse in enumerate(stat):
            x = np.arange(len(pulse))
            y = pulse
            ax[i].stackplot(x, y[:,0], y[:,1], y[:,2], labels=event_label)
            ax[i].set_title(f"Iso Sigma: {emitter_label[i]}")

        plt.subplots_adjust(hspace=0.5)
        ax[-2].legend(loc='upper center', bbox_to_anchor=(1.1, -0.2), 
                   fancybox=True, shadow=True, ncol=3)        
        fname = str(self.log_dir / "emitter_pulse.png")
        plt.savefig(fname,bbox_inches='tight')
        plt.clf(); plt.close()

    def archive_to_numpy(self, archive):
        grid_res = [len(a)-1 for a in archive.boundaries]
        n_beh    = archive._behavior_dim        

        if archive.use_objects:
            genome_archive = np.full(np.r_[grid_res, 1], np.nan, dtype=object)
        else:
            genome_archive = np.full(np.r_[grid_res, self.p['n_dof']], np.nan)
        fit_archive    = np.full(np.r_[grid_res, 1], np.nan)
        desc_archive   = np.full(np.r_[grid_res, n_beh], np.nan)
        meta_archive   = np.full(np.r_[grid_res, 1], np.nan, dtype=object)

        for elite in archive:
            fit_archive   [elite.idx[0], elite.idx[1],:] = elite.obj
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
        ax.set_title(self.p['exp_name'])
        plt.savefig(fname,bbox_inches='tight')
        plt.clf(); plt.close()

     

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
        for i, config_file in enumerate(self.p['config_files']):
            src = Path(config_file)
            dest = self.log_dir/f'{i}_{src.name}'
            shutil.copy(str(src), str(dest)) # for python <3.8

    def zip_results(self):
        file_name = Path(str(self.log_dir)+"_result")
        shutil.make_archive(file_name, 'zip', self.log_dir)



    # TODO: Move to logger
    # def as_numpy(self, include_metadata=False):
    #     # Create array
    #     grid_res = [len(a)-1 for a in self.boundaries]
    #     n_channels = sum([1, self._behavior_dim, self.p['n_dof']])
    #     np_archive = np.full(np.r_[grid_res, n_channels], np.nan)

    #     # Fill array
    #     # --> TODO: work on higher dim grids
    #     for elite in self:
    #         elite_stats = np.r_[elite.obj, elite.beh, elite.sol]
    #         np_archive[elite.idx[0], elite.idx[1], :] = elite_stats
    #     if not include_metadata:
    #         return np_archive
 