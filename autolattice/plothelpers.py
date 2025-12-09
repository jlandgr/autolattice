import numpy as np
import jax.numpy as jnp
import jax
from autolattice.jax_functions import dot_product
from itertools import product
from autolattice.analyse import extract_properties_of_interest
import matplotlib.pyplot as plt
import string
from autolattice.definitions import LEFT_TO_RIGHT, RIGHT_TO_LEFT

alphabet = string.ascii_lowercase

def prepare_plot_functions_for_plotting(optimizer, chain_length):
    dynamical_matrix_OBC = optimizer.__provide_function_for_dynamical_matrix__(chain_length, True)
    dynamical_matrix_PBC = optimizer.__provide_function_for_dynamical_matrix__(chain_length, False)
    scattering_func, info = optimizer.__provide_function_for_exact_scattering_calculation__(chain_length)
    scattering_func = jax.jit(jax.vmap(scattering_func, in_axes=[0,None]))
    eigsystem_function_transfer_matrix = jax.jit(jax.vmap(optimizer.calc_eigensystem_from_parameters, in_axes=[0,None]))
    scaling_func = jax.jit(jax.vmap(lambda a,b: optimizer.calc_gain_scaling(a,b, return_info=True), in_axes=[0,None]))
    scaling_func_reverse = jax.jit(jax.vmap(lambda a,b: optimizer.calc_gain_scaling(a,b, return_info=True, direction=RIGHT_TO_LEFT), in_axes=[0,None]))
    scaling_func_bulk_to_right = jax.jit(jax.vmap(lambda a,b: optimizer.calc_gain_scaling_bulk(a, b, direction=LEFT_TO_RIGHT), in_axes=[0,None]))
    scaling_func_bulk_to_left = jax.jit(jax.vmap(lambda a,b: optimizer.calc_gain_scaling_bulk(a, b, direction=RIGHT_TO_LEFT), in_axes=[0,None]))

    plot_functions = {
        'chain_length': chain_length,
        'modes_per_unit_cell': optimizer.modes_per_unit_cell,
        'dynamical_matrix_OBC': dynamical_matrix_OBC,
        'dynamical_matrix_PBC': dynamical_matrix_PBC,
        'scattering_func': scattering_func,
        'eigsystem_function_transfer_matrix': eigsystem_function_transfer_matrix,
        'scaling_func': scaling_func,
        'scaling_func_reverse': scaling_func_reverse,
        'scaling_func_bulk_to_right': scaling_func_bulk_to_right,
        'scaling_func_bulk_to_left': scaling_func_bulk_to_left,
    }

    return plot_functions

def plot_spectrum(ax, plot_functions, input_array):
    eigvals_OBC = jnp.linalg.eigvals(plot_functions['dynamical_matrix_OBC'](input_array))
    eigvals_PBC = jnp.linalg.eigvals(plot_functions['dynamical_matrix_PBC'](input_array))

    if np.max(np.real(eigvals_OBC)) > 0:
        print("unstable!")

    ax.plot(np.real(eigvals_OBC), np.imag(eigvals_OBC), label='OBC', marker='.', ls='None')
    ax.plot(np.real(eigvals_PBC), np.imag(eigvals_PBC), label='PBC', marker='.', ls='None')
    ax.axvline(0., color='black', ls='dashed')

    ax.set_xlabel('Re(eigenvalue)')
    ax.set_ylabel('Im(eigenvalue)')
    ax.legend()

def plot_scattering(axes, plot_functions, input_array, omegas, idxs_array):
    chain_length = plot_functions['chain_length']
    modes_per_unit_cell = plot_functions['modes_per_unit_cell']
    scattering = plot_functions['scattering_func'](omegas, input_array)[:,-modes_per_unit_cell:,:modes_per_unit_cell]
    prefactors, gain_rate, _ = plot_functions['scaling_func'](omegas, input_array)
    for idxs, ax in zip(idxs_array, axes):
        # ax.plot(omegas, jnp.log10(jnp.abs(scattering[:,idxs[0], idxs[1]])), label='scattering for N=%i'%chain_length)
        # ax.plot(omegas, jnp.log10(jnp.abs(prefactors[:,idxs[0], idxs[1]] * gain_rate ** (chain_length))), label='approximation')

        ax.plot(omegas, (jnp.abs(scattering[:,idxs[0], idxs[1]])), label='scattering for N=%i'%chain_length)
        ax.plot(omegas, (jnp.abs(prefactors[:,idxs[0], idxs[1]] * gain_rate ** (chain_length))), label='approximation', ls='dashed')
        
        # ax.plot(omegas, jnp.abs(prefactors[:,idxs[0], idxs[1]]), label='|prefactor|')
        ax.set_ylim(0,None)
        ax.set_xlabel('$\omega$')
        ax.set_ylabel('$(S_{1,N})_{%s,%s}$'%(alphabet[idxs[0]],alphabet[idxs[1]]))
    ax.legend()

def plot_scaling(ax, plot_functions, omegas, input_array, ylim, target_gain=None):
    if target_gain is not None:
        properties = extract_properties_of_interest(
            omegas=omegas,
            target_gain=target_gain,
            input_array=input_array,
            func_calculate_eigvals=plot_functions['eigsystem_function_transfer_matrix'],
            modes_per_unit_cell=plot_functions['modes_per_unit_cell']
        )

        for key in properties.keys():
            print(key, properties[key])
        
    eigvals_transfer_matrix, right_eigvectors_transfer_matrix, left_eigvectors_transfer_matrix, _ = plot_functions['eigsystem_function_transfer_matrix'](omegas, input_array)
    for idx in range(eigvals_transfer_matrix.shape[-1]):
        ax.plot(omegas, jnp.abs(eigvals_transfer_matrix[:,idx]))
    ax.set_ylim(ylim)
    ax.set_xlabel('$\omega$')
    ax.set_ylabel('scaling rate')

def plot_prefactors(ax, calc_scaling, omegas, input_array, modes_per_unit_cell):
    prefactors, gain_rate, _ = calc_scaling(omegas, input_array)
    # modes_per_unit_cell = prefactors.shape[1]
    for idx1 in range(modes_per_unit_cell):
        for idx2 in range(modes_per_unit_cell):
            ax.plot(omegas, np.abs(prefactors)[:,idx1, idx2], label=r'%s $\rightarrow$ %s'%(alphabet[idx1],alphabet[idx2]))
    ax.set_xlabel('$\omega$')
    ax.set_ylabel('|prefactor|')
    ax.legend()
    ax.set_ylim(0,None)

def plot_differences(ax, plot_functions, omegas, input_array):
    eigvals_transfer_matrix, right_eigvectors_transfer_matrix, left_eigvectors_transfer_matrix, _ = plot_functions['eigsystem_function_transfer_matrix'](omegas, input_array)
    modes_per_unit_cell = eigvals_transfer_matrix.shape[1]//2
    idx_dominant = modes_per_unit_cell-1
    ax.plot(omegas, np.abs(np.abs(eigvals_transfer_matrix[:,idx_dominant]) - np.abs(eigvals_transfer_matrix[:,idx_dominant-1])), label= r'$|\lambda _M| - |\lambda_{M-1}|$')
    ax.plot(omegas, np.abs(np.abs(eigvals_transfer_matrix[:,idx_dominant]) - np.abs(eigvals_transfer_matrix[:,idx_dominant+1])), label= r'$|\lambda _M| - |\lambda_{M+1}|$')

    ax.plot(omegas, np.abs(eigvals_transfer_matrix[:,idx_dominant] - eigvals_transfer_matrix[:,idx_dominant-1]), label= r'$|\lambda _M - \lambda_{M-1}|$')
    ax.plot(omegas, np.abs(eigvals_transfer_matrix[:,idx_dominant] - eigvals_transfer_matrix[:,idx_dominant+1]), label= r'$|\lambda _M - \lambda_{M+1}|$')

    eigvector_overlaps = jnp.abs(dot_product(right_eigvectors_transfer_matrix[:,:,idx_dominant], right_eigvectors_transfer_matrix[:,:,idx_dominant+1])/dot_product(right_eigvectors_transfer_matrix[:,:,idx_dominant], right_eigvectors_transfer_matrix[:,:,idx_dominant+1]))
    ax.plot(omegas, np.arccos(eigvector_overlaps), label=r'$\sphericalangle$ eigvecs')

    ax.set_ylim(0., 0.2)
    ax.set_xlabel('$\omega$')
    ax.set_ylabel('difference measures')
    ax.legend()

def create_fig(modes_per_unit_cell, plot_functions, omegas, input_array, target_gain=None, scaling_ylim=(1.2-0.2, 1.2+0.2)):

    # ncols = 2 + 2*modes_per_unit_cell
    # nrows = max(3, modes_per_unit_cell)

    ncols = 2 + modes_per_unit_cell
    nrows = max(2, modes_per_unit_cell)

    xsize = 3.5
    ysize = 2.5

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(xsize*ncols, ysize*nrows))

    plot_spectrum(axes[1,0], plot_functions, input_array)
    axes_scattering_1_N = axes[:,2:2+modes_per_unit_cell].flatten()
    idxs = [(idx1, idx2) for idx1, idx2 in product(range(modes_per_unit_cell),range(modes_per_unit_cell))]
    plot_scattering(axes_scattering_1_N, plot_functions, input_array, omegas, idxs)
    plot_scaling(axes[0,0], plot_functions, omegas, input_array, scaling_ylim, target_gain=target_gain)
    plot_prefactors(axes[0,1], plot_functions['scaling_func'], omegas, input_array, modes_per_unit_cell)
    plot_differences(axes[1,1], plot_functions, omegas, input_array)
    fig.tight_layout()
    return fig, axes

def calc_continuous_PBC(mus, ks=jnp.linspace(-np.pi, np.pi, 101)):
    mu_m1, mu_0, mu_p1 = mus
    H_k = jnp.exp(-1.j*ks)[:,None,None] * mu_m1[None] + mu_0[None] + jnp.exp(1.j*ks)[:,None,None] * mu_p1[None]
    eigvals = jnp.linalg.eigvals(H_k)
    #sort eigvals according to their real part
    eigvals_order = jnp.argsort(jnp.real(eigvals), 1)
    return ks, jnp.take_along_axis(eigvals, eigvals_order, 1)

def calc_det_H_k(mus, omega, ks=jnp.linspace(-np.pi, np.pi, 101)):
    mu_m1, mu_0, mu_p1 = mus
    num_modes = mu_0.shape[0]
    H_k = 1.j*(jnp.exp(-1.j*ks)[:,None,None] * mu_m1[None] + (mu_0[None] + 1.j*np.eye(num_modes) * omega) + jnp.exp(1.j*ks)[:,None,None] * mu_p1[None])
    return ks, jnp.linalg.det(H_k)