

[![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](./LICENSE) 

# bbq - Barely Bother Quality-Diversity
## Compact code and useful utilities for rapid testing w/QD

## Installation

Supported Python versions are 3.6 or later. For now just go in the base directory and:

```
$ pip install -e .

```
### Dependencies
- Numpy, pyribs, matplotlib etc. 
- Certainly more: when you get an error search on pip or conda, sorry.


## Usage

This library provides some ready made recipes for MAP-Elites using the pyRibs library to test out ideas with minimal setup and boilerplate. To get started just:

1. Define Problem Domain
   - Define a fitness (objective) function 
   - Define a descriptor (behavior/feature/measure) function
   - An expression function: how to go from floats between 0 and 1 to something your fitness and descriptor can evaluate. If it is only scaling this can be done define in the domain without explicitly rewriting the express function.
2. Set Hyperparameters
   - Just edit the .yaml file
3. Run MAP-Elites
   - To produce an archive run the `map_elites` function which takes a `domain` a hyperparameter dict `p`, and a `logger` that handles the results.

<details> 
<summary><b>1) Define Problem Domain</b></summary>

```python
""" Example using Rastrigin Function """

from ribs_helpers import RibsDomain # Base class for ribs domains

class Rastrigin(RibsDomain):
    def __init__(self, n_dof=10, n_desc=2, x_scale=[-5,5]):
        RibsDomain.__init__(self, n_dof, n_desc, x_scale)  
    
    def _fitness(self, x):        
        f = 10 * x.shape[0] + (x * x - 10 * np.cos(2 * math.pi * x)).sum()
        return -f + x.shape[0]**6 # Scale to make QD score increasing
    
    def _desc(self, x):
        return np.array(x[0:self.n_desc])
```
</details> 

<details> 
<summary><b>2) Set Hyperparameters</b></summary>

```yaml
name: 'rastrigin'

# -- Compute -- #
n_workers: 4

# -- Domain -- #
dof: 4
param_bounds: [-2, 2]
desc_bounds:
- [-2, 2]
- [-2, 2]
desc_labels:
- [Param 1]
- [Param 2]
iso_strength: 0.1

# -- MAP-Elites -- #
grid_res: [32, 32]
n_init: 512
n_gens: 50
n_emitters: 8
n_batch: 64

# -- Logging -- #
print_rate: 5 # Print to console
plot_rate: 5  # Plot graphs
save_rate: 50 # Save archive

```
</details> 



<details> 
<summary><b>3) Run MAP-Elite</b>s</summary>

```python
""" Running MAP-Elites with the Rastrigin Function """
import yaml
from map_elites import map_elites
from ribs_logger import RibsLogger
from domain_example import Rastrigin

def run_me(config_file='config.yaml'):
    p = yaml.load(open(config_file, "r"), Loader=yaml.FullLoader)
    domain = Rastrigin(n_dof=p['dof'], 
                       n_desc=len(p['desc_bounds']),
                       x_scale=p['param_bounds'])
    logger = RibsLogger(p, save_meta=True, copy_config=config_file, clear=False)
    archive = map_elites(domain, p, logger)    
    logger.zip_results()
```
</details> 

This sample code handles all logging, visualization, and parallelization.