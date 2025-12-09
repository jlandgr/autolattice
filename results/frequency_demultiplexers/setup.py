save_folder = 'frequency_splitter/test3'

target_scaling_rate = 1.1
omega_right = 3.
omega_left = -3.
constraints_scaling = [
    msc.Scaling_Rate_Constraint(target_scaling_rate, target_omega=omega_right),
    msc.Scaling_Rate_Constraint(0., gradient_order=1, target_omega=omega_right),
    msc.Scaling_Rate_Constraint(target_scaling_rate, target_omega=omega_left, direction=RIGHT_TO_LEFT),
    msc.Scaling_Rate_Constraint(0., gradient_order=1, target_omega=omega_left, direction=RIGHT_TO_LEFT),
    # msc.Scaling_Rate_Constraint(0., gradient_order=2, target_omega=0., kwargs_difference_quotient={'h': 1.e-4}),
    msc.Prefactor_Bulk_Constraint(0., gradient_order=1, target_omega=omega_right),
    msc.Prefactor_Bulk_Constraint(0., gradient_order=1, target_omega=omega_left, direction=RIGHT_TO_LEFT),
    msc.Min_Distance_Eigvals_range(jnp.linspace(-5., 5., 51), 0.05),
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