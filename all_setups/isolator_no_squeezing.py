# set the target characteristics for the isolotor and set up the optimizer
# this file will be executed by run_optimization.py

save_folder = 'isolators/without_squeezing'

target_scaling_rate = 1.
constraints_scaling = [
    msc.Scaling_Rate_Constraint(target_scaling_rate),  # enforce that the target scaling rate for amplification from left to right equals 1
    msc.Scaling_Rate_Constraint(0., gradient_order=1),  # enforce that its derivative equals zero
    msc.Scaling_Rate_Constraint(0., gradient_order=2),  # enforce that its second derivative equals zero
    msc.Prefactor_Constraint(1.),  # enforce that the prefactor of this scaling equals 1
    msc.Prefactor_Constraint(0., gradient_order=1), # enforce that its derivative equals zero
    msc.Min_Distance_Eigvals(min_distance=1.e-2)  # enforce that the central two eigenvalues have a specified minimum distance
    # msc.Stability_Constraint(10),  # all chains will be stable anyway as squeezing is switched of, so no stability constraint is required
]

optimizer = arch_opt.Architecture_Optimizer(
    2, # number of modes per unit cell
    mode_types=[True, True, True],  # defines which of the modes is a particle and which a hole, restricting the available couplings between the modes, see Methods A in the article for more details
    enforced_constraints=constraints_scaling,
    kappas_free_parameters=[False, True],  # loss rate of the first mode is always set to 1 and used for rescaling all other frequencies, the second loss rate is a free parameter left open for optimization
    kwargs_optimization = {'num_tests': 20}, # set hyperparameters for optimization
    pso_parameters={'max_iterations': 5000}
)
