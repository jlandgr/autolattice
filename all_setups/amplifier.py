# set the target characteristics for the amplifier and set up the optimizer
# this file will be executed by run_optimization.py

save_folder = 'amplifier'

target_scaling_rate = 1.2
constraints_scaling = [
    msc.Scaling_Rate_Constraint(target_scaling_rate),
    msc.Scaling_Rate_Constraint(0., gradient_order=1),
    msc.Scaling_Rate_Constraint(0., gradient_order=2),
    msc.Scaling_Rate_Constraint(0., gradient_order=3),
    msc.Prefactor_Constraint(0., gradient_order=1),
    msc.Min_Distance_Eigvals_range(jnp.linspace(-3., 3., 121), 0.05),
    msc.Stability_Constraint(10),
]

kwargs_bounds = {'bounds_extrinsic_loss': [0., np.inf]}

optimizer = arch_opt.Architecture_Optimizer(
    2,
    mode_types=[True, False, True],
    enforced_constraints=constraints_scaling,
    kappas_free_parameters=[False, True],
    port_intrinsic_losses=[False, False],
    kwargs_optimization = {'num_tests': 20, 'kwargs_bounds': kwargs_bounds},
    pso_parameters={'max_iterations': 5000}
)