import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import unittest

import jax
import copy
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import autolattice.scattering as mss
import autolattice.constraints as msc
import autolattice.architecture_optimizer as arch_opt
import autolattice.architecture as arch
import autolattice.jax_functions as msj

def fill_Hamiltonian_single_mode_per_unit_cell(params, chain_length):
    kappa0 = params['kappa_ext0']
    g00p = params['|g_{0,0p}|'] * jnp.exp(1.j*params[r'\mathrm{arg}(g_{0,0p})']) #* np.sqrt(kappa0*kappa0)
    Delta0 = params['Delta0'] #* kappa0

    system = mss.Multimode_system(chain_length)
    ext_losses = []
    for cell_idx in range(chain_length):
        mode0_idx = cell_idx
        mode0p_idx = cell_idx + 1

        ext_losses.extend([kappa0])
        system.add_adag_a_coupling(-Delta0, mode0_idx, mode0_idx)
        if cell_idx < chain_length - 1:
            system.add_adag_a_coupling(g00p, mode0_idx, mode0p_idx)
        
    system.ext_losses = jnp.array(ext_losses)
    return system

def fill_Hamiltonian_two_modes_per_unit_cell(params, chain_length):
    kappa0 = params['kappa_ext0']
    kappa1 = params['kappa_ext1']

    g00p = params['|g_{0,0p}|'] * jnp.exp(1.j*params[r'\mathrm{arg}(g_{0,0p})']) #* np.sqrt(kappa0*kappa0)
    g11p = params['|g_{1,1p}|'] * jnp.exp(1.j*params[r'\mathrm{arg}(g_{1,1p})']) #* np.sqrt(kappa1*kappa1)
    v01 = params[r'|\nu_{0,1}|'] * jnp.exp(1.j*params[r'\mathrm{arg}(\nu_{0,1})']) #* np.sqrt(kappa0*kappa1)
    v10p = params[r'|\nu_{1,0p}|'] * jnp.exp(1.j*params[r'\mathrm{arg}(\nu_{1,0p})']) #* np.sqrt(kappa1*kappa0)
    Delta0 = params['Delta0'] #* kappa0
    Delta1 = params['Delta1'] #* kappa1

    system = mss.Multimode_system(2*chain_length)
    ext_losses = []

    for cell_idx in range(chain_length):
        mode0_idx = 2*cell_idx
        mode1_idx = 2*cell_idx + 1
        mode0p_idx = 2*cell_idx + 2
        mode1p_idx = 2*cell_idx + 3

        ext_losses.extend([kappa0, kappa1])

        system.add_adag_a_coupling(-Delta0, mode0_idx, mode0_idx)
        system.add_adag_a_coupling(-Delta1, mode1_idx, mode1_idx)
        system.add_adag_adag_coupling(v01, mode0_idx, mode1_idx)
        if cell_idx < chain_length - 1:
            system.add_adag_a_coupling(g00p, mode0_idx, mode0p_idx)
            system.add_adag_adag_coupling(v10p, mode1_idx, mode0p_idx)
            system.add_adag_a_coupling(g11p, mode1_idx, mode1p_idx)

    system.ext_losses = jnp.array(ext_losses)

    return system

def return_selection(scattering_matrix, selected_modes):
    num_selected_modes = len(selected_modes)
    selection = np.zeros(list(scattering_matrix.shape[:-2]) + [num_selected_modes, num_selected_modes], scattering_matrix.dtype)
    
    for idx1 in range(num_selected_modes):
        for idx2 in range(num_selected_modes):
            selection[..., idx1, idx2] = scattering_matrix[..., selected_modes[idx1], selected_modes[idx2]]
    
    return selection

class Scattering_Test(unittest.TestCase):
    def test_exact_gain_calculation(self):
        optimizer = arch_opt.Architecture_Optimizer(
            2,
            mode_types=[True, False, True],
            make_initial_test=False,
            kappas_free_parameters=[False, True],
        )
        chain_length = 10
        omegas = jnp.linspace(-2., 2., 101)

        params_dict = {
            'Delta0': 0.0,
            'Delta1': 0.0,
            '|\\nu_{0,1}|': 0.36768538114394517,
            '|g_{0,0p}|': -0.962556666345482,
            '|\\nu_{0,1p}|': -0.4506238429793324,
            '|\\nu_{1,0p}|': 0.0,
            '|g_{1,1p}|': 0.795186892486225,
            '\\mathrm{arg}(\\nu_{0,1})': 1.5707316070320434,
            '\\mathrm{arg}(g_{0,0p})': 0.0,
            '\\mathrm{arg}(\\nu_{0,1p})': 0.0,
            '\\mathrm{arg}(\\nu_{1,0p})': 0.0,
            '\\mathrm{arg}(g_{1,1p})': 0.0,
            'kappa_ext1': 0.8485954565197984,
            'epsilon': 1.0
        }

        input_array = jnp.array([params_dict[var.__str__()] for var in optimizer.all_variables_list])

        scattering_func, info = optimizer.__provide_function_for_exact_scattering_calculation__(chain_length)
        scattering_func = jax.jit(jax.vmap(scattering_func, in_axes=[0,None]))
        S_plus = scattering_func(omegas, input_array)

        unit_cell_idx = chain_length - 3
        gain_exact_func = jax.jit(jax.vmap(lambda omegas, array: optimizer.calc_gain_exact(omegas, array, chain_length=chain_length, unit_cell_idx=unit_cell_idx, direction=arch_opt.RIGHT_TO_LEFT), in_axes=[0,None]))
        np.testing.assert_array_almost_equal(gain_exact_func(omegas, input_array)[:,1,0], S_plus[:,1,2*(unit_cell_idx-1)], 4)

        unit_cell_idx = chain_length
        gain_exact_func = jax.jit(jax.vmap(lambda omegas, array: optimizer.calc_gain_exact(omegas, array, chain_length=chain_length, unit_cell_idx=unit_cell_idx, direction=arch_opt.RIGHT_TO_LEFT), in_axes=[0,None]))
        np.testing.assert_array_almost_equal(gain_exact_func(omegas, input_array)[:,1,0], S_plus[:,1,2*(unit_cell_idx-1)], 4)

        unit_cell_idx = 1
        gain_exact_func = jax.jit(jax.vmap(lambda omegas, array: optimizer.calc_gain_exact(omegas, array, chain_length=chain_length, unit_cell_idx=unit_cell_idx, direction=arch_opt.RIGHT_TO_LEFT), in_axes=[0,None]))
        np.testing.assert_array_almost_equal(gain_exact_func(omegas, input_array)[:,1,0], S_plus[:,1,2*(unit_cell_idx-1)], 4)

    def test_single_mode_chain(self):
        optimizer = arch_opt.Architecture_Optimizer(
            1,
            mode_types=[True, True],
            make_initial_test=False,
            kappas_free_parameters=[True]
        )

        chain_length = 5
        omega = -2.
        params_dict = {
            'Delta0': 0.5,
            '|g_{0,0p}|': 1.5,
            r'\mathrm{arg}(g_{0,0p})': 0.3,
            'kappa_ext0': 2.,
            'epsilon': 1.
        }

        direction = arch_opt.RIGHT_TO_LEFT

        input_array = jnp.array([params_dict[var.__str__()] for var in optimizer.all_variables_list])
        system = fill_Hamiltonian_single_mode_per_unit_cell(params_dict, chain_length)
        np.testing.assert_array_almost_equal(optimizer.calc_gain_exact(omega, input_array, chain_length, direction=direction), system.return_scattering_matrix(jnp.array([omega]))[0,chain_length-1])

        chain_length = 10
        omega = -2.
        input_array = jnp.array([params_dict[var.__str__()] for var in optimizer.all_variables_list])
        system = fill_Hamiltonian_single_mode_per_unit_cell(params_dict, chain_length)
        np.testing.assert_array_almost_equal(optimizer.calc_gain_exact(omega, input_array, chain_length, direction=direction), system.return_scattering_matrix(jnp.array([omega]))[0,chain_length-1])


        chain_length = 20
        omega = -2.
        input_array = jnp.array([params_dict[var.__str__()] for var in optimizer.all_variables_list])
        system = fill_Hamiltonian_single_mode_per_unit_cell(params_dict, chain_length)
        np.testing.assert_array_almost_equal(optimizer.calc_gain_exact(omega, input_array, chain_length, direction=direction), system.return_scattering_matrix(jnp.array([omega]))[0,chain_length-1])

        chain_length = 10
        omega = -0.
        input_array = jnp.array([params_dict[var.__str__()] for var in optimizer.all_variables_list])
        system = fill_Hamiltonian_single_mode_per_unit_cell(params_dict, chain_length)
        np.testing.assert_array_almost_equal(optimizer.calc_gain_exact(omega, input_array, chain_length, direction=direction), system.return_scattering_matrix(jnp.array([omega]))[0,chain_length-1])

        chain_length = 10
        omega = -1.
        params_dict = {
            'Delta0': -0.8,
            '|g_{0,0p}|': 2.3,
            r'\mathrm{arg}(g_{0,0p})': -1.1,
            'kappa_ext0': 4.5,
            'epsilon': 1.
        }

        input_array = jnp.array([params_dict[var.__str__()] for var in optimizer.all_variables_list])
        system = fill_Hamiltonian_single_mode_per_unit_cell(params_dict, chain_length)
        np.testing.assert_array_almost_equal(optimizer.calc_gain_exact(omega, input_array, chain_length, direction=direction), system.return_scattering_matrix(jnp.array([omega]))[0,chain_length-1])
    
    def test_two_mode_chain(self):
        def return_scattering_matrix(params_dict, chain_length, omegas=jnp.array([0.])):
            selected_modes = []
            for idx in range(chain_length):
                selected_modes.append(2*idx)
                selected_modes.append(2*chain_length + 2*idx + 1)
                
            system = fill_Hamiltonian_two_modes_per_unit_cell(params_dict, chain_length)
            return return_selection(system.return_scattering_matrix(omegas), selected_modes)
        
        optimizer = arch_opt.Architecture_Optimizer(
            2,
            mode_types=[True, False, True],
            make_initial_test=False,
            enforced_constraints=[msc.Constraint_coupling_zero(0,'1p',2)],
            kappas_free_parameters=[True, True]
        )

        params_dict = {
            'Delta0': 0.5,
            'Delta1': -0.8,
            '|g_{0,0p}|': 1.,
            '|g_{1,1p}|': 1.4,
            r'|\nu_{0,1}|': 0.26,
            r'|\nu_{1,0p}|': 0.53,
            r'\mathrm{arg}(g_{0,0p})': 0.5,
            r'\mathrm{arg}(g_{1,1p})': -np.pi,
            r'\mathrm{arg}(\nu_{0,1})': 0.2,
            r'\mathrm{arg}(\nu_{1,0p})': -0.3,
            'kappa_ext0': 1.87,
            'kappa_ext1': 2.56,
            'epsilon': 1.,
        }
        input_array = jnp.array([params_dict[var.__str__()] for var in optimizer.all_variables_list])
        direction = arch_opt.RIGHT_TO_LEFT
        # optimizer.calc_gain_rate(0.5, input_array, 10)

        chain_lengths = [10, 20]
        omegas = jnp.linspace(-.3,.3, 51)
        for chain_length in chain_lengths:
            scattering_matrix = return_scattering_matrix(params_dict, chain_length, omegas)[:,:2,-2:]
            
            # ax.plot(omegas, target_element/np.max(target_element), label='%i'%chain_length)

            expected_gain = []
            for omega in omegas:
                expected_gain.append(optimizer.calc_gain_exact(omega, input_array, chain_length, direction=direction))
            expected_gain = jnp.array(expected_gain)

            for i in range(2):
                for j in range(2):
                    np.testing.assert_array_almost_equal(scattering_matrix[:,i,j], expected_gain[:,i,j])

    def test_eigvalue_eigvectors(self):
        matrix_dimension = 8
        matrix_power = 3
        test_matrix = np.random.uniform(-5., 5., (matrix_dimension, matrix_dimension))
        product_matrix = jnp.linalg.matrix_power(test_matrix, matrix_power)

        eigvals, right_eigvectors = jnp.linalg.eig(test_matrix)
        left_eigvectors = jnp.linalg.inv(right_eigvectors)
        # ith right eigenvector is right_eigvectors[:,i]
        # ith left eigenvector is right_eigvectors[i,:]?

        approximate_matrix = jnp.zeros((matrix_dimension, matrix_dimension))
        for idx in range(matrix_dimension):
            eigval = eigvals[idx]
            right_eigvec = right_eigvectors[:,idx]
            left_eigvec = left_eigvectors[idx,:]
            approximate_matrix += eigval**matrix_power * jnp.outer(right_eigvec, left_eigvec)
        np.testing.assert_array_almost_equal(approximate_matrix, product_matrix)

    def test_noise_on_resonance(self, direction=arch_opt.RIGHT_TO_LEFT):
        params_dict = {'Delta0': 0.0,
                'Delta1': 0.0,
                '|\\nu_{0,1}|': 0.42418861879122577,
                '|g_{0,0p}|': 1.255902202652076,
                '|\\nu_{0,1p}|': -0.4393631447898572,
                '|\\nu_{1,0p}|': 0.0,
                '|g_{1,1p}|': -0.7714885726485206,
                '\\mathrm{arg}(\\nu_{0,1})': -1.5707927354398636,
                '\\mathrm{arg}(g_{0,0p})': 0.,
                '\\mathrm{arg}(\\nu_{0,1p})': 0.0,
                '\\mathrm{arg}(\\nu_{1,0p})': 0.0,
                '\\mathrm{arg}(g_{1,1p})': 0.0,
                'kappa_ext0': 1.,
                'kappa_ext1': 1.138657169120285,
                'epsilon': 1.0}
        
        chain_length = 40
        
        optimizer = arch_opt.Architecture_Optimizer(
            2,
            mode_types=[True, False, True],
            kappas_free_parameters=[False, True],
            make_initial_test=False
        )

        if direction == arch_opt.LEFT_TO_RIGHT:
            params_dict['\\mathrm{arg}(\\nu_{0,1})'] = - params_dict['\\mathrm{arg}(\\nu_{0,1})']

        input_array = jnp.array([params_dict[var.__str__()] for var in optimizer.all_variables_list])
        scattering_func, info = optimizer.__provide_function_for_exact_scattering_calculation__(chain_length)
        S_plus = scattering_func(0., input_array)
        S_minus = S_plus

        for i in range(2):
            for j in range(2):
                idxs = (i,j)

                if direction == arch_opt.RIGHT_TO_LEFT:
                    scattering = S_plus[:optimizer.modes_per_unit_cell, -optimizer.modes_per_unit_cell:]
                    total_noise = 1/4 * jnp.sum((jnp.abs(S_plus)**2 + jnp.abs(S_minus)**2)[idxs[0]], -1)
                else:
                    scattering = S_plus[-optimizer.modes_per_unit_cell:,:optimizer.modes_per_unit_cell]
                    total_noise = 1/4 * jnp.sum((jnp.abs(S_plus)**2 + jnp.abs(S_minus)**2)[2*(chain_length-1) + idxs[0]], -1)

                quadrature = jnp.angle(scattering[idxs[0], idxs[1]])
                gain = 1/4*jnp.abs(scattering[idxs[0], idxs[1]] * np.exp(-1.j*quadrature) + jnp.conjugate(scattering[idxs[0], idxs[1]]) * np.exp(1.j*quadrature))**2

                excact_added_noise = total_noise/gain-1/2
                approximation = optimizer.calc_output_noise_on_resonance(input_array, output_idx=idxs[0], input_idx=idxs[1], bath_occupations=jnp.zeros(2), amplification_direction=direction)

                np.testing.assert_array_almost_equal(excact_added_noise, approximation, decimal=4)
    
    def test_noise_on_resonance2(self, direction=arch_opt.RIGHT_TO_LEFT):

        optimizer = arch_opt.Architecture_Optimizer(
            2,
            mode_types=[True, False, True],
            make_initial_test=False,
            enforced_constraints=[],
            kappas_free_parameters=[False, True],
            gradient_method=arch_opt.DIFFERENCE_QUOTIENT,
            kwargs_optimization = {'num_tests': 10}
        )

        kappa_ext1 = 2.268125463872672
        params_dict = {
            'Delta0': 0.11121739888667385,
            'Delta1': 0.7355598195845932 * kappa_ext1,
            '|\\nu_{0,1}|': 0.9162912108597017 * jnp.sqrt(kappa_ext1),
            '|g_{0,0p}|': 0.09394887525996315,
            '|\\nu_{0,1p}|': 0.6190552562559497 * jnp.sqrt(kappa_ext1),
            '|\\nu_{1,0p}|': -0.21799292957620248 * jnp.sqrt(kappa_ext1),
            '|g_{1,1p}|': 1.4084234843650385 * kappa_ext1,
            '\\mathrm{arg}(\\nu_{0,1})': 2.158059454246464,
            '\\mathrm{arg}(g_{0,0p})': 1.1826408001656257,
            '\\mathrm{arg}(\\nu_{0,1p})': 1.8398157722003654,
            '\\mathrm{arg}(\\nu_{1,0p})': -0.4260072756387134,
            '\\mathrm{arg}(g_{1,1p})': 2.0737090123554522,
            'kappa_ext1': kappa_ext1,
            'epsilon': 1.0
        }
        if direction == arch_opt.LEFT_TO_RIGHT:
            raise NotImplementedError()

        input_array = jnp.array([params_dict[var.__str__()] for var in optimizer.all_variables_list])

        np.testing.assert_array_almost_equal(optimizer.calc_output_noise_on_resonance(input_array, 0, 0, bath_occupations=np.zeros(2), amplification_direction=direction), 0.5, decimal=4)

    def test_noise_on_resonance_left_to_right(self):
        self.test_noise_on_resonance(arch_opt.LEFT_TO_RIGHT)

    def test_scaling(self, direction=arch_opt.RIGHT_TO_LEFT):
        kappa_ext1 = 0.8485954565197984
        input_dict = {
            'Delta0': 0.0,
            'Delta1': 0.0,
            '|\\nu_{0,1}|': 0.36768538114394517 * jnp.sqrt(kappa_ext1),
            '|g_{0,0p}|': -0.962556666345482,
            '|\\nu_{0,1p}|': -0.4506238429793324 * jnp.sqrt(kappa_ext1),
            '|\\nu_{1,0p}|': 0.0 * jnp.sqrt(kappa_ext1),
            '|g_{1,1p}|': 0.795186892486225 * kappa_ext1,
            '\\mathrm{arg}(\\nu_{0,1})': 1.5707316070320434,
            '\\mathrm{arg}(g_{0,0p})': 0.0,
            '\\mathrm{arg}(\\nu_{0,1p})': 0.0,
            '\\mathrm{arg}(\\nu_{1,0p})': 0.0,
            '\\mathrm{arg}(g_{1,1p})': 0.0,
            'kappa_ext1': kappa_ext1,
            'epsilon': 1.0
        }

        if direction == arch_opt.LEFT_TO_RIGHT:
            input_dict['\\mathrm{arg}(\\nu_{0,1})'] = - input_dict['\\mathrm{arg}(\\nu_{0,1})']

        omegas = jnp.linspace(-1., 1., 31)
        chain_length = 30

        optimizer = arch_opt.Architecture_Optimizer(
            2,
            mode_types=[True, False, True],
            kappas_free_parameters=[False, True],
            make_initial_test=False
        )

        input_array = jnp.array([input_dict[var.__str__()] for var in optimizer.all_variables_list])
        scaling_func = jax.jit(jax.vmap(lambda omegas, input_array: optimizer.calc_gain_approximate(omegas, input_array, chain_length=chain_length, direction=direction), in_axes=[0,None]))
        approximation = np.log10(np.abs(scaling_func(omegas, input_array)))[:,0,0]

        scattering_func, info = optimizer.__provide_function_for_exact_scattering_calculation__(chain_length)
        scattering_func = jax.jit(jax.vmap(scattering_func, in_axes=[0,None]))
        full_scattering_matrix = scattering_func(omegas, input_array)
        if direction == arch_opt.LEFT_TO_RIGHT:
            gain_exact = np.log10(np.abs(full_scattering_matrix))[:,2*(chain_length-1)+0,0]
        elif direction == arch_opt.RIGHT_TO_LEFT:
            gain_exact = np.log10(np.abs(full_scattering_matrix))[:,0,2*(chain_length-1)+0]

        np.testing.assert_array_almost_equal(np.round(approximation), np.round(gain_exact))

    def test_scaling_left_to_right(self):
        self.test_scaling(arch_opt.LEFT_TO_RIGHT)


    def test_scaling_for_intermediate_unit_cells(self):
        input_dict = {
            'Delta0': 0.0,
            'Delta1': 0.0,
            '|\\nu_{0,1}|': 0.36768538114394517,
            '|g_{0,0p}|': -0.962556666345482,
            '|\\nu_{0,1p}|': -0.4506238429793324,
            '|\\nu_{1,0p}|': 0.0,
            '|g_{1,1p}|': 0.795186892486225,
            '\\mathrm{arg}(\\nu_{0,1})': 1.5707316070320434,
            '\\mathrm{arg}(g_{0,0p})': 0.0,
            '\\mathrm{arg}(\\nu_{0,1p})': 0.0,
            '\\mathrm{arg}(\\nu_{1,0p})': 0.0,
            '\\mathrm{arg}(g_{1,1p})': 0.0,
            'kappa_ext1': 0.8485954565197984,
            'epsilon': 1.0
        }

        omegas = jnp.linspace(-1., 1., 31)
        unit_cell_idx = 29
        idx1 = 0
        idx2 = 0
        direction = arch_opt.RIGHT_TO_LEFT

        optimizer = arch_opt.Architecture_Optimizer(
            2,
            mode_types=[True, False, True],
            kappas_free_parameters=[False, True],
            make_initial_test=False
        )

        input_array = jnp.array([input_dict[var.__str__()] for var in optimizer.all_variables_list])
        scaling_func = jax.jit(jax.vmap(lambda omegas, input_array: optimizer.calc_gain_scaling(omegas, input_array, return_info=True, direction=direction), in_axes=[0,None]))
        prefactors, scaling_rate, infos = scaling_func(omegas, input_array)
        chain_length = 30

        projector = optimizer.give_projector(direction)
        Xs = [projector@u_v@(projector.T) for u_v in infos['outer_products']]

        correction = np.zeros([len(omegas), 2,2], dtype='complex')
        for considered_order in [1,2,3]:
            correction += Xs[considered_order] * infos['eigvals'][:,considered_order,None,None]**(chain_length-unit_cell_idx)
        approximation_improved = - (jnp.sqrt(infos['kappa_ext_matrix_unit_cell']) @ infos['approximated_inverse_prefactor'] @ correction @ infos['mu1_inv'] @ jnp.sqrt(infos['kappa_ext_matrix_unit_cell']))[:,idx1,idx2] * scaling_rate**chain_length

        scattering_func, info = optimizer.__provide_function_for_exact_scattering_calculation__(chain_length)
        scattering_func = jax.jit(jax.vmap(scattering_func, in_axes=[0,None]))
        full_scattering_matrix = scattering_func(omegas, input_array)
        gain_exact = np.log10(np.abs(full_scattering_matrix))[:,idx1,2*(unit_cell_idx-1)+idx2]
        
        np.testing.assert_array_almost_equal(np.round(np.log10(np.abs(approximation_improved))), np.round(gain_exact))

    def test_scaling_for_intermediate_unit_cells_new(self):
        direction = arch_opt.RIGHT_TO_LEFT
        
        input_dict = {
            'Delta0': 0.0,
            'Delta1': 0.0,
            '|\\nu_{0,1}|': 0.36768538114394517,
            '|g_{0,0p}|': -0.962556666345482,
            '|\\nu_{0,1p}|': -0.4506238429793324,
            '|\\nu_{1,0p}|': 0.0,
            '|g_{1,1p}|': 0.795186892486225,
            '\\mathrm{arg}(\\nu_{0,1})': 1.5707316070320434,
            '\\mathrm{arg}(g_{0,0p})': 0.0,
            '\\mathrm{arg}(\\nu_{0,1p})': 0.0,
            '\\mathrm{arg}(\\nu_{1,0p})': 0.0,
            '\\mathrm{arg}(g_{1,1p})': 0.0,
            'kappa_ext1': 0.8485954565197984,
            'epsilon': 1.0
        }

        omegas = jnp.linspace(-1., 1., 31)
        unit_cell_idx = 29
        idx1 = 0
        idx2 = 0
        direction = arch_opt.RIGHT_TO_LEFT
        chain_length = 30

        optimizer = arch_opt.Architecture_Optimizer(
            2,
            mode_types=[True, False, True],
            kappas_free_parameters=[False, True],
            make_initial_test=False
        )
        input_array = jnp.array([input_dict[var.__str__()] for var in optimizer.all_variables_list])

        scattering_func, info = optimizer.__provide_function_for_exact_scattering_calculation__(chain_length)
        scattering_func = jax.jit(jax.vmap(scattering_func, in_axes=[0,None]))

        calc_approximation = jax.jit(jax.vmap(lambda omega, array: optimizer.calc_gain_approximation_intermediate_mode(omega, array, chain_length, unit_cell_idx, direction=direction), in_axes=[0,None]))
        full_scattering_matrix = scattering_func(omegas, input_array)
        gain_exact = np.log10(np.abs(full_scattering_matrix))[:,idx1,2*(unit_cell_idx-1)+idx2]
        approximation = calc_approximation(omegas, input_array)
        
        np.testing.assert_array_almost_equal(np.round(np.log10(np.abs(approximation)))[:,idx1, idx2], np.round(gain_exact))

    def test_noise_on_input(self):
        kappa_ext1 = 1.138657169120285
        input_dict = {'Delta0': 0.0,
        'Delta1': 0.0,
        '|\\nu_{0,1}|': 0.42418861879122577 * jnp.sqrt(kappa_ext1),
        '|g_{0,0p}|': 1.255902202652076,
        '|\\nu_{0,1p}|': -0.4393631447898572 * jnp.sqrt(kappa_ext1),
        '|\\nu_{1,0p}|': 0.0 * jnp.sqrt(kappa_ext1),
        '|g_{1,1p}|': -0.7714885726485206 * kappa_ext1,
        '\\mathrm{arg}(\\nu_{0,1})': 1.5707927354398636,
        '\\mathrm{arg}(g_{0,0p})': 0.,
        '\\mathrm{arg}(\\nu_{0,1p})': 0.0,
        '\\mathrm{arg}(\\nu_{1,0p})': 0.0,
        '\\mathrm{arg}(g_{1,1p})': 0.0,
        'kappa_ext0': 1.,
        'kappa_ext1': kappa_ext1,
        'epsilon': 1.0}

        optimizer = arch_opt.Architecture_Optimizer(
            2,
            mode_types=[True, False, True],
            kappas_free_parameters=[False, True],
            make_initial_test=False
        )
        input_array = jnp.array([input_dict[var.__str__()] for var in optimizer.all_variables_list])

        approximation = optimizer.calc_input_noise_on_resonance(input_array, 0, amplification_direction=arch_opt.LEFT_TO_RIGHT)
        np.testing.assert_almost_equal(approximation, 0.63736819)

        approximation = optimizer.calc_input_noise_on_resonance(input_array, 1, amplification_direction=arch_opt.LEFT_TO_RIGHT)
        np.testing.assert_almost_equal(approximation, 0.54271981)

    def test_input_reflection(self):
        nice_solution = np.array([0., 0., 0.31960394, -0.6430136, -0.36568645, 0., 0.74675939, 0., 0., np.pi/2., 0., 0.])
        chain_length = 50
        omegas = jnp.linspace(-3., 3., 101)
        optimizer = arch_opt.Architecture_Optimizer(
            2,
            mode_types=[True, False, True],
            make_initial_test=False,
        )
        scattering_func, info = optimizer.__provide_function_for_exact_scattering_calculation__(chain_length)
        scattering_func = jax.jit(jax.vmap(scattering_func, in_axes=[0,None]))
        scattering = scattering_func(omegas, nice_solution)
        for input_idx in range(2):
            approximation = []
            for omega in omegas:
                approximation.append(optimizer.calc_reflection_at_input(omega, nice_solution, input_idx))
            approximation = np.array(approximation)
            np.testing.assert_array_almost_equal(approximation, scattering[:,input_idx, input_idx], decimal=2)

    def test_compare_to_Claras_chain(self):
        constraints_scaling = [
            msc.Constraint_coupling_zero('0', '1p', 2),
            msc.Constraint_coupling_zero('1', '1p', 2),
            msc.Constraint_coupling_phase_zero(0, '0p', 2),
            msc.Constraint_coupling_phase_zero(1, '0p', 2),
            msc.Constraint_coupling_zero(0, 0, 2),
            msc.Constraint_coupling_zero(1, 1, 2)
        ]

        optimizer = arch_opt.Architecture_Optimizer(
            2,
            mode_types=[True, True, True],
            make_initial_test=False,
            enforced_constraints=constraints_scaling,
            kwargs_optimization = {'num_tests': 10, 'interrupt_if_successful': True},
            port_intrinsic_losses=[True, False],
            kappas_free_parameters=[True, True]
        )
        eigsystem_function_transfer_matrix = jax.jit(jax.vmap(optimizer.calc_eigensystem_from_parameters, in_axes=[0,None]))

        kappa1 = 10000.

        gamma = 0.25
        theta = np.pi/2
        kappa = 0.4
        Gamma = 0.32
        # J = 0.3
        # Lambda = 4 * J / (gamma + 2*Gamma - kappa)
        Lambda = 2
        J = Lambda/4 * (gamma + 2*Gamma - kappa)
        C = 2 * Gamma / (gamma + 2*Gamma - kappa)

        gabs = np.sqrt(Gamma * kappa1/4)
        subs_dict = {
            optimizer.__init_gabs__(0,2): J,
            optimizer.__init_gabs__(0,1): gabs,
            optimizer.__init_gabs__(1,2): gabs,
            optimizer.__init_gphase__(0,1): - theta,
            optimizer.variables_intrinsic_losses[0]: -kappa,
            optimizer.variables_kappas_ext[1]: kappa1,
            optimizer.variables_kappas_ext[0]: gamma,
        }

        input_array = jnp.array([subs_dict[var] for var in optimizer.all_variables_list])

        omegas = jnp.linspace(-1., 1., 101)

        mu_m1 = - (1.j * J + Gamma * jnp.exp(1.j*theta)/2)
        mu_0 = (kappa - gamma - 2 * Gamma) / 2 + 1.j*omegas
        mu_1 = - (1.j * J + Gamma * jnp.exp(-1.j*theta)/2)

        z_plus = (-mu_0 + np.sqrt(mu_0**2 - 4*mu_1 * mu_m1))/(2*mu_1)
        z_minus = (-mu_0 - np.sqrt(mu_0**2 - 4*mu_1 * mu_m1))/(2*mu_1)

        eigvals_transfer_matrix, _, _, _ = eigsystem_function_transfer_matrix(omegas, input_array)

        # test full spectrum, deviations get larger far away from omega=0
        np.testing.assert_array_almost_equal(eigvals_transfer_matrix[:,1], z_minus, decimal=4)
        np.testing.assert_array_almost_equal(eigvals_transfer_matrix[:,2], z_plus, decimal=2)

        # at omega=0 deviations should be zero, independent of the choice of kappa1
        np.testing.assert_array_almost_equal(eigvals_transfer_matrix[len(omegas)//2,1], z_minus[len(omegas)//2])
        np.testing.assert_array_almost_equal(eigvals_transfer_matrix[len(omegas)//2,2], z_plus[len(omegas)//2])


class Jax_function_tests(unittest.TestCase):
    def test_adjugate_matrix(self):
        np.random.seed(0)
        matrix = jnp.array(np.random.uniform(-5., 5., (6,6)) + 1.j*np.random.uniform(-5., 5., (6,6)))
        adjugate_matrix = msj.adjugate(matrix)
        np.testing.assert_array_almost_equal(adjugate_matrix, jnp.linalg.inv(matrix)*jnp.linalg.det(matrix))