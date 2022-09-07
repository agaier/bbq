import yaml

def load_config(config_files):
    """ Combines yaml files into single configuration dict 
    *Note*: Later fields overwrite earlier
    """
    p = {}
    for file in config_files:
        yaml_dict = yaml.load(open(file, "r"), Loader=yaml.FullLoader)
        p = {**p, **yaml_dict}
    p['config_files'] = config_files
    return p

def scale(x, param_scale):
    """ Scale each column of matrix by mins and maxes defined in vectors"""
    v_min, v_max = param_scale[0], param_scale[1]
    v_range = v_max-v_min
    v_scaled = (v_range * x) + v_min
    return v_scaled    
