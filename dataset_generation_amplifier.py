'''
This python script generates a large dataset of parameter sets that realize the amplifier behavior defined in the file all_setups/amplifier.py and discussed in Fig. 3 in our article
The lattice structure is fixed to the lattice shown in Fig. 3(b).
We vary the amplification rate per unit cell (target_scaling_rate) and the script finds suitable coupling rates to achieve this amplification rate.
The corresponding parameters are saved and later merged into one file by dataset_merge.ipynb.
This dataset is used by AIFeynman to find parameter dependencies between the transport properties and the coupling parameters.

We recommend to run this script on a cluster to parallelize the dataset generation.
We assume in the following that the cluster uses Slurm as job scheduler.
'''

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import os
import jax.numpy as jnp
import pickle
import autolattice.constraints as msc
import autolattice.architecture_optimizer as arch_opt

import copy

# the environment variable SLURM_PROCID tells the script which is its current task number.
# if you want to run this script locally on your computer, you can set SLURM_PROCID to 0 by executing the following command in the UNIX command line:
# export SLURM_PROCID=0
procid = int(os.environ['SLURM_PROCID'])

num_runs = 10000

# fix the lattice structure the one shown in Fig. 3 (b)
graph_to_test = np.array([0, 2, 1, 0, 0, 2, 1], dtype='int')

save_folder = 'amplifier_dataset/'

for num_run in range(num_runs):

    # define the target behavior
    target_scaling_rate = np.random.uniform(1., 3.)  # set randomly a target scaling rate per unit cell
    # same constraints as shown in all_setups/amplifier.py
    constraints_scaling = [
        msc.Scaling_Rate_Constraint(target_scaling_rate),
        msc.Scaling_Rate_Constraint(0., gradient_order=1),
        msc.Scaling_Rate_Constraint(0., gradient_order=2),
        msc.Scaling_Rate_Constraint(0., gradient_order=3),
        msc.Prefactor_Constraint(0., gradient_order=1),
        msc.Min_Distance_Eigvals_range(jnp.linspace(-3., 3., 121), 0.05),
        msc.Stability_Constraint(10),
    ]

    # set up the optimizer
    kwargs_bounds = {'bounds_extrinsic_loss': [0., np.inf]}
    optimizer = arch_opt.Architecture_Optimizer(
        2,
        mode_types=[True, False, True],
        enforced_constraints=constraints_scaling,
        kappas_free_parameters=[False, False],
        port_intrinsic_losses=[False, False],
        kwargs_optimization = {'num_tests': 20, 'kwargs_bounds': kwargs_bounds},
        pso_parameters={'max_iterations': 5000}
    )

    success_mod, info_mod, success_idxs = optimizer.repeated_optimization(graph_characteriser=graph_to_test)
    print(num_run, success_mod)
    if success_mod:
        to_save_dict = copy.deepcopy(info_mod[-1]['solution_dict'])

        filename = os.path.join(save_folder, 'procid' + str(procid).zfill(5) + ',run=' + str(num_run).zfill(5) + '.pkl')

        with open(filename, 'wb') as f:
            pickle.dump(to_save_dict, f)