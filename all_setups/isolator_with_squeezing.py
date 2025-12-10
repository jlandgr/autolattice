# same as isolator_no_squeezing, but this time squeezing is allowed
# this file will be executed by run_optimization.py

save_folder = 'isolators/with_squeezing'

target_scaling_rate = 1.
constraints_scaling = [
    msc.Scaling_Rate_Constraint(target_scaling_rate),
    msc.Scaling_Rate_Constraint(0., gradient_order=1),
    msc.Scaling_Rate_Constraint(0., gradient_order=2),
    msc.Scaling_Rate_Constraint(0., gradient_order=3),
    msc.Prefactor_Constraint(1.),
    msc.Prefactor_Constraint(0., gradient_order=1),
    msc.Prefactor_Constraint(0., gradient_order=2),
    msc.Min_Distance_Eigvals_range(jnp.linspace(-5., 5., 51), 0.1),
    msc.Stability_Constraint(10),  # enforce that a chain of 10 unit cells is stable
]

kwargs_bounds = {'bounds_extrinsic_loss': [0., np.inf]}

optimizer = arch_opt.Architecture_Optimizer(
    2,
    mode_types=[True, False, True],  # defines which of the modes is a particle and which a hole, restricting the available couplings between the modes, see Methods A in the article for more details
    enforced_constraints=constraints_scaling,
    kappas_free_parameters=[False, True],
    kwargs_optimization = {'num_tests': 20, 'kwargs_bounds': kwargs_bounds},
    pso_parameters={'max_iterations': 5000}
)
