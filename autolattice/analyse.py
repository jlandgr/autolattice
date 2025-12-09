import jax.numpy as jnp
import numpy as np

def extract_properties_of_interest(omegas, target_gain, input_array, func_calculate_eigvals, modes_per_unit_cell, error_tolerance_bandwidth=1.e-2):
    idx_of_interest = modes_per_unit_cell - 1 # Mth largest eigenvalue determines scaling of gain rate
    eigvals_transfer_matrix, _, _, _ = func_calculate_eigvals(omegas, input_array)
    eigvals_abs = jnp.abs(eigvals_transfer_matrix)

    properties_of_interest = {}
    
    # calculate_bandwidth
    idxs_within_tolerance = np.where(np.abs(eigvals_abs[:,idx_of_interest]-eigvals_abs[len(omegas)//2,idx_of_interest]) < error_tolerance_bandwidth)[0]
    left_cutoff = idxs_within_tolerance[0]
    right_cutoff = idxs_within_tolerance[-1]

    omega_left = omegas[left_cutoff]
    omega_right = omegas[right_cutoff]
    properties_of_interest['bandwidth'] = omega_right - omega_left

    # calculate minimal distance to next larger eigenvalue within the bandwidth
    distance_to_next_larger_eigenvalue = np.abs(eigvals_abs[:,idx_of_interest+1] - eigvals_abs[:,idx_of_interest])
    min_distance_next_eigenvalue = np.min(distance_to_next_larger_eigenvalue[left_cutoff:right_cutoff+1])

    properties_of_interest['min_distance_next_eigenvalue'] = min_distance_next_eigenvalue
    properties_of_interest['min_ratio_to_next_larger_eigenvalue'] = (target_gain+min_distance_next_eigenvalue)/target_gain

    distance_at_zero = distance_to_next_larger_eigenvalue[len(omegas)//2]
    properties_of_interest['distance_next_eigenvalue_at_0'] = distance_at_zero
    properties_of_interest['ratio_to_next_larger_eigenvalue_at_zero'] = (target_gain+distance_at_zero)/target_gain

    return properties_of_interest