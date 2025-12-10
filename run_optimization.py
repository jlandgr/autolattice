'''
This python script discovers all lattice models fulfilling the target transport characteristics specified in a setup file.
We recommend to run this script on a cluster to parallelize the search throught the discrete search space.
We assume in the following that the cluster uses Slurm as job scheduler.
'''

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import os
import autolattice.constraints as msc
import autolattice.architecture_optimizer as arch_opt

from autolattice.definitions import LEFT_TO_RIGHT, RIGHT_TO_LEFT

# the environment variable SLURM_NTASKS stores how many total tasks Slurm has allocated for your job, we recommend using around 100 to distribute the work load
# the environment variable SLURM_PROCID tells the script which is its current task number.
# if you want to run this script locally on your computer, you can set these variables by using, e.g.:
# export SLURM_PROCID=0
# export SLURM_NTASKS=500
num_tasks = int(os.getenv('SLURM_NTASKS'))
procid = int(os.getenv('SLURM_PROCID'))

# load the specified target characteristics from  a specified setup file and set ups the optimizer class
# see the folder all_setups for the target characteristics used in our work
# we recommend to look at isolator_no_squeezing.py for as an introductory example
setup_file = 'all_setups/isolator_no_squeezing.py'
# setup_file = 'all_setups/isolator_with_squeezing.py'
# setup_file = 'all_setups/amplifier.py'
# setup_file = 'all_setups/frequency_demultiplexer.py'

with open(setup_file) as f:
    code = f.read()
    print(code)
    exec(code)

# create results folder
os.makedirs(save_folder, exist_ok=True)
# copy the setup file to the corresponding save folder
if procid == 0:
    os.system("cp %s %s"%(setup_file, os.path.join(save_folder, 'setup.py')))  

# create a list of all possible lattice structures
optimizer.prepare_all_possible_combinations()

# divide this list into the subset which has to be tested by this task
full_division = np.array_split(optimizer.list_possible_graphs, num_tasks)
my_set_to_test = full_division[procid]

solution_arrays = []
successes = []

# test the respective subset
print('testing %i out of %i graphs'%(len(my_set_to_test), len(optimizer.list_possible_graphs)))
for graph_to_test in my_set_to_test:
    success, infos, _ = optimizer.repeated_optimization(graph_characteriser=graph_to_test)
    solution_arrays.append(infos[-1]['solution'])
    successes.append(success)
    print('tested graph:', graph_to_test, 'success:', success)

# save the results
to_save = {
    'graphs_tested': my_set_to_test,
    'success': successes,
    'solutions': solution_arrays
}
np.savez(os.path.join(save_folder, str(procid).zfill(5) + '.npz'), **to_save)