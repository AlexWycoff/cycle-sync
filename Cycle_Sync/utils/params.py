def parse_params(params):
    defaults = {
        'tolQuad': 1e-10,
        'delt': 1e-8,
        'maxit': 200,
        'maxitQuad': 200,
        'staglim': 5,
        'sinmin': 0.6,
        'tau4': 4,
        'flam': lambda x: x / (x + 10),
        'tau1': 20,
        'WLSiters': 20,
        'beta': 20,
        'nsample': 200,
        'aab_iters': 5,
        'maxtri': 100,
        'ridge': 1e-10,
        'seed': None
    }
    for k, v in defaults.items():
        if k not in params:
            params[k] = v
    return params
