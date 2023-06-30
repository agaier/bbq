

[![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](./LICENSE) 

![](assets/icon_1.png)

# BBQ. A Tasty Front-End for PyRibs.

* [Purpose](#purpose)
* [Installation](#installation)
* [Workflow](#workflow)
* [Problem Domains](#problem-domains)
* [Configuration Files](#configuration-files)
* [Visualization](#visualization)

## Purpose
BBQ is a collection of scripts designed to reduce the amount of boilerplate needed for setting up new MAP-Elites domains with the PyRibs library without extensive knowledge of PyRibs or MAP-Elites. The core of BBQ is:

- **Domain class:**  The only python code necessary to write is the optimization domain
- **Configuration:** All algorithm settings are stored in a yaml file, allowing easy changes in a central human readable format, and for simpler handling of experiments
- **Visualization:** Visualization of key metrics is automatically performed during a run, and visualization functions are provided for more indepth exploration in notebooks. 

## Installation
![](assets/icon_2.png)

**NOTE**: This is based on pyribs 0.4 -- the current version is 0.5. Many non-backwards compatibile bits have been added
(renaming core parameters and such), and lot of functionality that is included here (batch addition, dumping archive) is
now in core pyribs. It looks great! When I have time I will move this over and take advantage of all of their hard work.

Supported Python versions are 3.6 or later. Just go in the base directory and:

```
$ pip install -e .

```

To make sure everything works:

```
python tests/rast_test.py
```

A `log` directory should be created from this brief run filled with plots, metrics, recorded archives, and the configuration files used in the run.

## Workflow

![](assets/icon_3.png)

This library provides some ready made recipes for MAP-Elites using the pyRibs library to test out ideas with minimal setup and boilerplate. To get started just:

1. **Define Problem Domain**
   - *Fitness function* (objective) 
   - *Descriptor function *(behavior/feature/measure) 
   - *Expression function:* how to go from floats between 0 and 1 to something your fitness and descriptor can evaluate. If the encoding and mutation is more complicated than that an individual can be defined as object instead (see here).
2. **Set Hyperparameters**
   - Just edit the .yaml file
3. **Run MAP-Elites** 
   - ```archive = map_elites(domain, p, logger)```
   - Takes problem `domain` a hyperparameter dict `p`, and a `logger` that handles the results.
4. **Visualize Results** 
   - A log file is created during a run that is populated with data and charts, replicates put each in a folder
   - Notebook and functions are provided for [exploring single archives](notebooks/archive_exploration.ipynb), visualizing [summary results and comparing algorithms](notebooks/summary_results.ipynb)

### Problem Domains
The simplest definition of domain can be created by defining only a fitness function and a descriptor function. This assumes that all genomes are 0-1 scaled to a predefined range. The output of the expressed genome is saved as metadata. To add other values to the metadata the inherited `evaluate` parent function will have to be rewritten.

If the hyperparameter for `n_workers` equal 1 batch evaluation is done with a list comprehension, if greater than 1 a dask instance will be created and used to evaluate in parallel.

An example is here: `bbq/examples/rastrigin.py`

```python
class Rastrigin(BbqDomain):
    def __init__(self, param_bounds=[-5.12, 5.12], **kwargs):
        self.param_bounds = param_bounds
        BbqDomain.__init__(self, **kwargs)        
    
    def _fitness(self, x):        
        f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
        return -f + 2*x.shape[0]**2 # Shift to make QD score increasing    
        
    def _desc(self, x):
        return np.array(x[0:2])

    def express(self, x):
        return scale(x, self.param_bounds)
```
   

### Configuration Files

To speed up running experiments, replicates, and tweaking hyperparameters all settings of an experiment are kept in a human readable configuration file and passed into the code. A sample configuration file looks like this:

```yaml
task_name: 'rastrigin'

# -- Optimization -- #
n_init: 100
n_gens: 500

# -- Parameters -- #
n_dof: 10             # Dimensions in Rastrigin
param_bounds: [-2, 2] # Overwrite domain default

# -- Archive -- #
archive:
  type: "Grid"
  grid_res: [20,20]
  desc_bounds:
  - [-2, 2]
  - [-2, 2]
  desc_labels:
  - Param 1
  - Param 2  

# -- Emitters -- #
emitters:
  -
    name: "Gaussian1"
    type: "Gauss"
    batch_size: 25
    sigma0: 0.01
  -
    name: "Gaussian2"
    type: "Gauss"
    batch_size: 25
    sigma0: 0.05    


# -- Logging -- #
print_rate: 5  # Print to console
plot_rate: 25  # Plot graphs
save_rate: 50  # Save archive
```

To run an experiment comparing these settings, the updated fields can be placed in an additional yaml file and loaded to overwrite those fields. Replacing the gaussian emitters with a CMA-ME improvement emitter and a Line emitter in the `config/line_cma_mix.yaml` file.

```yaml
# -- Emitters -- #
exp_name: 'line_cma_mix'

emitters:
  -
    name: "CMA-1"
    type: "Cma"
    batch_size: 25
    sigma0: 0.005
  -
    name: "Line1"
    type: "Line"
    batch_size: 25
    iso_sigma: 0.005
    line_sigma: 0.1
```

Then run  ```python rast_test.py config/line_cma_mix.yaml```

## Visualization
![](assets/icon_4.png)


- For single archives see [this notebook](notebooks/archive_exploration.ipynb)
- To view summary results use [this notebook](notebooks/summary_results.ipynb)
