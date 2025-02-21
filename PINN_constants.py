#%%
import numpy as np
import optax
import PINN_domain, PINN_trackdata, PINN_network, PINN_problem
import os
import shutil
import pickle
from pathlib import Path
class ConstantsBase:
    def __getitem__(self, key):
        if key not in vars(self): raise KeyError(f'key "{key}" not defined in class')
        return getattr(self, key)
    def __setitem__(self, key, item):
        if key not in vars(self): raise KeyError(f'key "{key}" not defined in class')
        setattr(self, key, item)
    def __str__(self):
        s = repr(self) + '\n'
        for k in vars(self): s+=f"{k}: {self[k]}\n"
        return s
    
    @property
    def summary_out_dir(self):
        return f"/scratch/hyun/results/summaries/{self.run}/"
    @property
    def model_out_dir(self):
        return f"/scratch/hyun/results/models/{self.run}/"

    def get_outdirs(self):
        if os.path.exists(self.summary_out_dir):
            print('Loading saved checkpoints')
            #shutil.rmtree(self.summary_out_dir)
        else:
            Path(self.summary_out_dir).mkdir(exist_ok=True, parents=True)
        if os.path.exists(self.model_out_dir):
            print('Loading saved checkpoints')
            #shutil.rmtree(self.model_out_dir)
        else:
            Path(self.model_out_dir).mkdir(exist_ok=True, parents=True)


    def save_constants_file(self):
        with open(self.summary_out_dir + f"constants_{self.run}.txt", 'w') as f:
            for k in vars(self): f.write(f"{k}: {self[k]}\n")
        with open(self.summary_out_dir + f"constants_{self.run}.pickle", 'wb') as f:
            pickle.dump(vars(self), f)
    @property
    def constants_file(self):
        return self.summary_out_dir + f"constants_{self.run}.pickle"

def print_c_dicts(c_dicts):
    keys = []
    for c_dict in c_dicts[::-1]:
        for k in c_dict.keys():
            if k not in keys: keys.append(k)

    for k in keys:
        print(f"{k}: ",end="")
        for i,c_dict in enumerate(c_dicts):
            if k in c_dict.keys(): item=str(c_dict[k])
            else: item='None'
            if i == len(c_dicts)-1: print(f"{item}",end="")
            else: print(f"{item} | ",end="")
        print("")    

class Constants(ConstantsBase):
    def __init__(self, **kwargs):
        self.run = "HIT"

        
        self.domain_init_kwargs = dict(domain_range = {'t':(0,0.1),'x':(0,0.1),
                                                       'y':(0,0.1),'z':(0,0.1)},
                                       frequency = 1000, grid_size = [9, 200, 200, 200],
                                       bound_keys = [''])

        
        self.data_init_kwargs = dict(path = '', domain_range = {'t':(0,0.1),'x':(0,0.1),
                                     'y':(0,0.1),'z':(0,0.1)}, timeskip = 1,
                                      track_limit = 100000, frequency = 1000, data_keys = ['pos', 'vel'], viscosity = 15*10**(-6))

        self.network_init_kwargs = dict(layer_sizes = [4, 16, 32, 16, 4], network_name = 'MLP')

        self.problem_init_kwargs = dict(domain_range = {'t':(0,0.1),'x':(0,0.1),
                                     'y':(0,0.1),'z':(0,0.1)}, viscosity = 15e-6,
                                     loss_weights = (1,1,1,0.00001,0.00001,0.00001,0.00001),
                                     path_s = '/home/bussard/hyun_sh/TBL_PINN/data/HIT/IsoturbFlow.mat',
                                     frequency = 1250, constraints = ('first_order_diff', 'second_order_diff', 'second_order_diff', 'second_order_diff'),
                                     problem_name = 'HIT')

        self.optimization_init_kwargs = dict(optimiser = '', learning_rate = 1e-3,
                                             n_steps = 30000, p_batch = 5000,
                                             e_batch = 5000, b_batch = 5000)
        
        for key in kwargs.keys(): self[key] = kwargs[key]

        self.domain = PINN_domain.Domain
        self.data = PINN_trackdata.Data
        self.network = eval('PINN_network.'+ self.network_init_kwargs['network_name'])
        self.problem = eval('PINN_problem.'+self.problem_init_kwargs['problem_name'])
if __name__ == "__main__":
    from PINN_domain import *
    from PINN_trackdata import *
    from PINN_network import *

    run = "run00"
    all_params = {"domain":{}, "data":{}, "network":{}}

    # Set Domain params
    frequency = 23900
    domain_range = {'t':(0,50/frequency), 'x':(0,0.058), 'y':(0,0.00321), 'z':(0,0.0171)}
    grid_size = [51, 2800, 600, 212]
    bound_keys = ['ic', 'bcxu', 'bcxl', 'bcyu', 'bcyl', 'bczu', 'bczl']
    

    # Set Data params
    path = '/RealWall/to_distribute/for_FlowFit_0.8/'
    timeskip = 1
    track_limit = 424070
    data_keys = ['pos', 'vel', 'acc', ]
    viscosity = 15*10**(-6)
    
    # Set network params
    key = random.PRNGKey(1)
    layer_sizes = [4, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 4]
    network_name = 'MLP'


    # Set problem params
    path_s = '/scratch/hyun//RealWall/to_distribute/validate_02/'
    problem_name = 'Rwall'
    loss_weights = (1.0, 1.0, 1.0, 0.0000001, 0.00000001, 0.00000001, 0.00000001, 1.0)
    constraints = ('first_order_diff', 'second_order_diff', 'second_order_diff', 'second_order_diff')
    # Set optimization params
    n_steps = 100000
    optimiser = optax.adam
    learning_rate = 1e-3
    p_batch = 5000
    e_batch = 5000
    b_batch = 5000


    c = Constants(
        run= run,
        domain_init_kwargs = dict(domain_range = domain_range, frequency = frequency, 
                                  grid_size = grid_size, bound_keys=bound_keys),
        data_init_kwargs = dict(path = path, domain_range = domain_range, timeskip = timeskip,
                                track_limit = track_limit, frequency = frequency, data_keys = data_keys),
        network_init_kwargs = dict(key = key, layer_sizes = layer_sizes, network_name = network_name),
        problem_init_kwargs = dict(domain_range = domain_range, viscosity = viscosity, loss_weights = loss_weights,
                                   path_s = path_s, frequency = frequency, problem_name = problem_name),        
        optimization_init_kwargs = dict(optimiser = optimiser, learning_rate = learning_rate, n_steps = n_steps,
                                        p_batch = p_batch, e_batch = e_batch, b_batch = b_batch)
    )

    c.get_outdirs()
    c.save_constants_file()

# %%
