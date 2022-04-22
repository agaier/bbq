import yaml

def create_config(base_file, exp_file=None):
    """ Combines yaml files into single configuration dict """
    p = yaml.load(open(base_file, "r"), Loader=yaml.FullLoader)
    if exp_file is not None:
        exp_config  = yaml.load(open(exp_file, "r"), Loader=yaml.FullLoader)
        p = {**p, **exp_config}
        p['exp_config_path'] = exp_file
    p['base_config_path'] = base_file
    return p


def scale(x, param_scale):
    """ Scale each column of matrix by mins and maxes defined in vectors"""
    v_min, v_max = param_scale[0], param_scale[1]
    v_range = v_max-v_min
    v_scaled = (v_range * x) + v_min
    return v_scaled    
