

def me_benchmark(Problem, task, p, name='tdomino', rep=None):
    d = Problem(task, p)
    p['alg_name'] = name
    logger = Logger(p, clear=False, rep=rep) # <--TODO: copy config as yaml
    archive = map_elites(d, p, logger, emitter_type=IsoLineEmitter, archive_type=d.archive_type)
    print("Done")

if __name__ == '__main__':
    from tdomino.bench_algs import NSGA2, ME_Single, ME_Sum, TDomino
    from bench_problems import get_obj_fcn

    # -- MOO Rastrigin -- #
    base_config = 'config/rast.yaml'   

    # -- Experiment Settings -- #
    exp_config = 'config/test.yaml'
    #exp_config = 'config/smoke.yaml'

    exp_config = 'config/test.yaml'
    #exp_config = 'config/smoke.yaml'

    p = create_config(base_config, exp_config)
    task = get_obj_fcn(p['task_name'])

    me_benchmark(TDomino,   task, p, name='tdomino')  