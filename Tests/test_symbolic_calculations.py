import unittest
import numpy as np
import sympy as sp

import jax
jax.config.update("jax_enable_x64", True)

from autolattice.symbolic import setup_conditions, setup_conditions_subset, extend_matrix, find_numerical_solution_real_cs, extract_coupling_rates, symbolic_solver, Mode, setup_Bogoliubov_conditions, initialize_dynamical_matrix, symbolically_calc_scattering_matrix, setup_conditions_dynamical_to_scattering, setup_unitary_conditions
from autolattice.scattering import calc_scattering_matrix, Multimode_system

class Symbolic_transformation_Tests_forward_direction(unittest.TestCase):
    def test_link_two_inputs_to_one_output_and_isolate(self):
        a = sp.Symbol('a', real=True)
        S_target = sp.Matrix([[0,0,0],[0,0,0],[a,a,0]])
        paras, S_extended = extend_matrix(S_target, 5, real_cs=True)

        operators = [Mode().a for _ in range(5)]
        dynamical_matrix, _ = initialize_dynamical_matrix(operators, real=True)
        S_actual = symbolically_calc_scattering_matrix(dynamical_matrix)

        conditions_unifier = setup_conditions_dynamical_to_scattering(dynamical_matrix, S_extended)
        conditions_unifier += [S_actual[2,0] - a, S_actual[2,1] - a]

        conditions_old, _ = setup_conditions(S_extended, S_upper_left=True)
        
        conditions_unitary_right, _ = setup_unitary_conditions(S_extended.T)
        solutions_right = symbolic_solver(conditions_unitary_right, paras+[a], dict=True)

        subs_dict = {paras[8]: 1, paras[5]: 1}
        np.testing.assert_array_equal(np.zeros(15), [sp.simplify(c.subs(solutions_right[0]).subs(subs_dict)) for c in conditions_old])

        to_invert = S_extended.subs(solutions_right[0]).subs(subs_dict) - sp.eye(5)
        inverse = to_invert.adjugate()/to_invert.det()

        subs_dynamical = {}
        for idx1 in range(5):
            for idx2 in range(idx1):
                subs_dynamical[dynamical_matrix[idx2, idx1]] = sp.simplify(inverse[idx2,idx1])

        np.testing.assert_array_equal(np.zeros(9), [sp.simplify(c.subs(subs_dynamical)).subs(a, -sp.sqrt(sp.S(1)/2)) for c in conditions_unifier])

class Symbolic_transformation_Tests_subset_conditions(unittest.TestCase):
    def test_circulator(self):
        S_target = sp.Matrix([[0,-1,0],[0,0,-1],[-1,0,0]])
        modes = [Mode() for idx in range(3)]
        operators = [m.a for m in modes]
        conditions, _ = setup_conditions_subset(operators, S_target)
        conditions_evaluated = [float(condition) for condition in conditions]
        np.testing.assert_array_equal(np.zeros(6), conditions_evaluated)

    def test_isolator_2_modes(self):
        S_target = sp.Matrix([[0,-1],[0,0]])
        modes = [Mode() for idx in range(2)]
        operators = [m.a for m in modes]
        conditions, _ = setup_conditions_subset(operators, S_target)
        conditions_evaluated = [float(condition) for condition in conditions]
        np.testing.assert_array_almost_equal([-0.5, -0.5, 1.], conditions_evaluated)

    def test_isolator_2_modes_1_added_known_solution(self):
        S_target = sp.Matrix([[0,-1],[0,0]])
        free_parameters, S_extended = extend_matrix(S_target, 3, real_cs=True)
        modes = [Mode() for idx in range(3)]
        operators = [m.a for m in modes]
        conditions, _ = setup_conditions_subset(operators, S_extended)
        simple_solution = {
            free_parameters[0]: 0,
            free_parameters[1]: -1,
            free_parameters[2]: -1,
            free_parameters[3]: 0,
            free_parameters[4]: 0
        }

        conditions_evaluated = [float(condition.subs(simple_solution)) for condition in conditions]
        np.testing.assert_array_equal(np.zeros(6), conditions_evaluated)

    def test_link_two_inputs_to_one_output_and_isolate(self):
        expected_solution = np.array([-1/2, -1/2, -1/2, 0, np.sqrt(2)/2, np.sqrt(2)/2])

        S_target = sp.sqrt(2)/2*sp.Matrix([[0,0,0],[0,0,0],[1,1,0]])
        modes = [Mode() for idx in range(3)]
        operators = [m.a for m in modes]
        conditions, _ = setup_conditions_subset(operators, S_target)
        conditions_evaluated = [float(condition) for condition in conditions]
        np.testing.assert_array_almost_equal(expected_solution, conditions_evaluated)

    def test_3_mode_directional_amplifier(self):
        # see PHYS. REV. APPLIED 7, 024028 (2017)
        expected_solution = np.zeros(6)

        g = sp.Symbol('g', real=True, positive=True)
        S_target = sp.Matrix([[0,0,1],[sp.sqrt(g),sp.sqrt(g+1),0],[sp.sqrt(g+1),sp.sqrt(g),0]])
        modes = [Mode() for idx in range(3)]
        operators = [modes[0].a, modes[1].adag, modes[2].a]
        conditions, _ = setup_conditions_subset(operators, S_target)
        conditions = [sp.simplify(cond) for cond in conditions]
        conditions_evaluated = [float(condition) for condition in conditions]

        np.testing.assert_array_almost_equal(expected_solution, conditions_evaluated)

    def test_4_mode_directional_trans_amplifier(self):
        # 4 mode directional amplifier, see arXiv:2305.04184
        expected_solution = np.zeros(10)

        g = sp.Symbol('g', real=True, positive=True)
        S_target = sp.Matrix([[0,0,1,0],[sp.sqrt(g+1),0,0,sp.sqrt(g)],[0,1,0,0],[sp.sqrt(g),0,0,sp.sqrt(g+1)]])
        modes = [Mode() for idx in range(4)]
        operators = [modes[0].a, modes[1].a, modes[2].a, modes[3].adag]
        conditions, info = setup_conditions_subset(operators, S_target)
        conditions = [sp.simplify(cond) for cond in conditions]
        conditions_evaluated = [float(condition) for condition in conditions]

        np.testing.assert_array_almost_equal(expected_solution, conditions_evaluated)

    def test_4_mode_directional_cis_amplifier(self):
        # 4 mode directional amplifier, see arXiv:2305.04184
        expected_solution = np.zeros(10)

        g = sp.Symbol('g', real=True, positive=True)
        S_target = sp.Matrix([[0,0,1,0],[sp.sqrt(g),0,0,sp.sqrt(g+1)],[sp.sqrt(g+1),0,0,sp.sqrt(g)],[0,1,0,0]])
        modes = [Mode() for idx in range(4)]
        operators = [modes[0].a, modes[1].adag, modes[2].a, modes[3].adag]
        conditions, info = setup_conditions_subset(operators, S_target)
        conditions = [sp.simplify(cond) for cond in conditions]
        conditions_evaluated = [float(condition) for condition in conditions]
        
        np.testing.assert_array_almost_equal(expected_solution, conditions_evaluated)

class Symbolic_Transformation_Tests_full_conditions(unittest.TestCase):
    def test_circulator(self):
        S_target = sp.Matrix([[0,-1,0],[0,0,-1],[-1,0,0]])
        conditions, _ = setup_conditions(S_target, S_upper_left=True)
        conditions_evaluated = [float(condition) for condition in conditions]
        np.testing.assert_array_equal(np.zeros(6), conditions_evaluated)

    def test_isolator_2_modes(self):
        S_target = sp.Matrix([[0,-1],[0,0]])
        conditions, _ = setup_conditions(S_target, S_upper_left=True)
        conditions_evaluated = [float(condition) for condition in conditions]
        np.testing.assert_array_almost_equal([-0.5, -0.5, 1.], conditions_evaluated)
            
    def test_isolator_2_modes_1_added_known_solution(self):
        S_target = sp.Matrix([[0,-1],[0,0]])
        free_parameters, S_extended = extend_matrix(S_target, 3, real_cs=True)
        conditions, _ = setup_conditions(S_extended, S_upper_left=True)
        simple_solution = {
            free_parameters[0]: 0,
            free_parameters[1]: -1,
            free_parameters[2]: -1,
            free_parameters[3]: 0,
            free_parameters[4]: 0
        }

        conditions_evaluated = [float(condition.subs(simple_solution)) for condition in conditions]
        np.testing.assert_array_equal(np.zeros(6), conditions_evaluated)

    def test_isolator_2_modes_1_added_find_solutions(self):
        expected_solutions = np.asarray([(0, -1, -1, 0, 0), (0, 1, 1, 0, 0)])
        
        S_target = sp.Matrix([[0,-1],[0,0]])
        free_parameters, S_extended = extend_matrix(S_target, 3, real_cs=True)
        conditions, _ = setup_conditions(S_extended, S_upper_left=True)
        solutions = sp.solve(conditions, free_parameters)

        np.testing.assert_equal(expected_solutions, np.asarray(solutions))

    def test_link_two_inputs_to_one_output_and_isolate(self):
        expected_solution = np.array([-1/2, -1/2, -1/2, 0, np.sqrt(2)/2, np.sqrt(2)/2])

        S_target = sp.sqrt(2)/2*sp.Matrix([[0,0,0],[0,0,0],[1,1,0]])
        conditions, _ = setup_conditions(S_target, S_upper_left=True)
        conditions_evaluated = [float(condition) for condition in conditions]
        np.testing.assert_array_almost_equal(expected_solution, conditions_evaluated)

    def test_link_two_inputs_to_one_output_and_isolate_add_1_mode(self):
        S_target = sp.sqrt(2)/2*sp.Matrix([[0,0,0],[0,0,0],[1,1,0]])
        free_parameters, S_extended = extend_matrix(S_target, 4, real_cs=True)
        conditions, _ = setup_conditions(S_extended, S_upper_left=True)
        solutions = sp.solve(conditions, free_parameters)
        self.assertEqual(0, len(solutions))

    def test_own_symbolic_solver_for_equ_system_without_solution(self):
        a, b, c = sp.symbols('a b c')
        # system of equation should have no solutions
        self.assertEqual(0, len(symbolic_solver([c, a+b],[a,b])))

    def test_own_symbolic_solver_for_equ_system_with_solution(self):
        a, b, c = sp.symbols('a b c')
        # system of equation should have solutions
        self.assertEqual(1, len(symbolic_solver([a+b],[a,b])))

    def test_unidirectional_amplifier_3_modes(self):
        # Compare own code with arxiv 2207.13728

        N_modes = 3

        J_abs = 1.
        J_phase = np.pi / 2.
        gc = 0.6 * J_abs
        gs = gc
        kappa_e = 2.6 * J_abs
        Delta = 0.

        system = Multimode_system(N_modes)
        for mode in range(N_modes):
            system.add_adag_a_coupling(-Delta, mode, mode)
            system.add_adag_adag_coupling(-gs, mode, mode)
        
        for mode in range(N_modes-1):
            system.add_adag_a_coupling(J_abs*np.exp(-1.j*J_phase), mode, mode+1)
            system.add_adag_adag_coupling(-gc, mode, mode+1)

        system.ext_losses = kappa_e * np.ones(N_modes)

        scattering_matrix, _ = calc_scattering_matrix(np.array([0.]), *system.return_system_parameters())

        conditions, _ = setup_conditions(scattering_matrix[0], symbolic=False)

        # num_conditions = conditions for diagonal + symmetry if A + As identical + B symmetric + Bs identical
        num_conditions = N_modes + (N_modes**2-N_modes)//2 + N_modes**2 + (N_modes**2-N_modes)//2 + N_modes**2
        expected_outcome = np.zeros(num_conditions)

        np.testing.assert_array_almost_equal(conditions, expected_outcome)

        modes = [Mode() for _ in range(N_modes)]
        operators = [m.a for m in modes] + [m.adag for m in modes]
        conditions_Bogoliubov, _ = setup_Bogoliubov_conditions(operators, sp.Matrix(scattering_matrix[0]))
        num_dimensions = 2*N_modes
        np.testing.assert_array_almost_equal(np.asarray(conditions_Bogoliubov, dtype='complex'), np.zeros(num_dimensions+(num_dimensions**2-num_dimensions)//2))

    def test_unidirectional_amplifier_5_modes(self):
        # Compare own code with arxiv 2207.13728

        N_modes = 5

        J_abs = 1.
        J_phase = np.pi / 2.
        gc = 0.6 * J_abs
        gs = gc
        kappa_e = 2.6 * J_abs
        Delta = 0.

        system = Multimode_system(N_modes)
        for mode in range(N_modes):
            system.add_adag_a_coupling(-Delta, mode, mode)
            system.add_adag_adag_coupling(-gs, mode, mode)
        
        for mode in range(N_modes-1):
            system.add_adag_a_coupling(J_abs*np.exp(-1.j*J_phase), mode, mode+1)
            system.add_adag_adag_coupling(-gc, mode, mode+1)

        system.ext_losses = kappa_e * np.ones(N_modes)

        scattering_matrix, _ = calc_scattering_matrix(np.array([0.]), *system.return_system_parameters())

        conditions, _ = setup_conditions(scattering_matrix[0], symbolic=False)

        # num_conditions = conditions for diagonal + symmetry if A + As identical + B symmetric + Bs identical
        num_conditions = N_modes + (N_modes**2-N_modes)//2 + N_modes**2 + (N_modes**2-N_modes)//2 + N_modes**2
        expected_outcome = np.zeros(num_conditions)

        np.testing.assert_array_almost_equal(conditions, expected_outcome)

        modes = [Mode() for _ in range(N_modes)]
        operators = [m.a for m in modes] + [m.adag for m in modes]
        conditions_Bogoliubov, _ = setup_Bogoliubov_conditions(operators, sp.Matrix(scattering_matrix[0]))
        num_dimensions = 2*N_modes
        np.testing.assert_array_almost_equal(np.asarray(conditions_Bogoliubov, dtype='complex'), np.zeros(num_dimensions+(num_dimensions**2-num_dimensions)//2))

    def test_unidirectional_amplifier_10_modes(self):
        # Compare own code with arxiv 2207.13728

        N_modes = 10

        J_abs = 1.
        J_phase = np.pi / 2.
        gc = 0.6 * J_abs
        gs = gc
        kappa_e = 2.6 * J_abs
        Delta = 0.

        system = Multimode_system(N_modes)
        for mode in range(N_modes):
            system.add_adag_a_coupling(-Delta, mode, mode)
            system.add_adag_adag_coupling(-gs, mode, mode)
        
        for mode in range(N_modes-1):
            system.add_adag_a_coupling(J_abs*np.exp(-1.j*J_phase), mode, mode+1)
            system.add_adag_adag_coupling(-gc, mode, mode+1)

        system.ext_losses = kappa_e * np.ones(N_modes)

        scattering_matrix, _ = calc_scattering_matrix(np.array([0.]), *system.return_system_parameters())

        conditions, _ = setup_conditions(scattering_matrix[0], symbolic=False)

        # num_conditions = conditions for diagonal + symmetry if A + As identical + B symmetric + Bs identical
        num_conditions = N_modes + (N_modes**2-N_modes)//2 + N_modes**2 + (N_modes**2-N_modes)//2 + N_modes**2
        expected_outcome = np.zeros(num_conditions)

        np.testing.assert_array_almost_equal(conditions, expected_outcome)

        modes = [Mode() for _ in range(N_modes)]
        operators = [m.a for m in modes] + [m.adag for m in modes]
        conditions_Bogoliubov, _ = setup_Bogoliubov_conditions(operators, sp.Matrix(scattering_matrix[0]))
        num_dimensions = 2*N_modes
        np.testing.assert_array_almost_equal(np.asarray(conditions_Bogoliubov, dtype='complex'), np.zeros(num_dimensions+(num_dimensions**2-num_dimensions)//2))

    def test_unidirectional_amplifier_20_modes(self):
        # Compare own code with arxiv 2207.13728

        N_modes = 20

        J_abs = 1.
        J_phase = np.pi / 2.
        gc = 0.6 * J_abs
        gs = gc
        kappa_e = 2.6 * J_abs
        Delta = 0.

        system = Multimode_system(N_modes)
        for mode in range(N_modes):
            system.add_adag_a_coupling(-Delta, mode, mode)
            system.add_adag_adag_coupling(-gs, mode, mode)
        
        for mode in range(N_modes-1):
            system.add_adag_a_coupling(J_abs*np.exp(-1.j*J_phase), mode, mode+1)
            system.add_adag_adag_coupling(-gc, mode, mode+1)

        system.ext_losses = kappa_e * np.ones(N_modes)

        scattering_matrix, _ = calc_scattering_matrix(np.array([0.]), *system.return_system_parameters())

        conditions, _ = setup_conditions(scattering_matrix[0], symbolic=False, conditions_using_adjugate=False)

        # num_conditions = conditions for diagonal + symmetry if A + As identical + B symmetric + Bs identical
        num_conditions = N_modes + (N_modes**2-N_modes)//2 + N_modes**2 + (N_modes**2-N_modes)//2 + N_modes**2

        np.testing.assert_array_almost_equal(conditions, np.zeros(num_conditions))

        modes = [Mode() for _ in range(N_modes)]
        operators = [m.a for m in modes] + [m.adag for m in modes]
        conditions_Bogoliubov, _ = setup_Bogoliubov_conditions(operators, sp.Matrix(scattering_matrix[0]), symbolic=False)
        num_dimensions = 2*N_modes
        np.testing.assert_array_almost_equal(np.asarray(conditions_Bogoliubov, dtype='complex'), np.zeros(num_dimensions+(num_dimensions**2-num_dimensions)//2))

    def test_system_with_random_couplings_including_squeezing(self):
        np.random.seed(0)
        N_modes = 5
        J_lim = 1.
        g_lim = 0.5
        system = Multimode_system(N_modes)

        detunings = []
        for idx in range(N_modes):
            coupling = np.random.uniform(-J_lim, J_lim)
            detunings.append(coupling)
            system.add_adag_a_coupling(coupling, idx, idx)

        J_couplings = []
        for idx2 in range(N_modes):
            for idx1 in range(idx2):
                coupling = np.random.uniform(-J_lim, J_lim) * np.exp(1.j*np.random.uniform(-np.pi, np.pi))
                J_couplings.append(coupling)
                system.add_adag_a_coupling(coupling, idx1, idx2)
        
        squeezing_interactions = []
        for idx2 in range(N_modes):
            for idx1 in range(idx2+1):
                coupling = np.random.uniform(-g_lim, g_lim) * np.exp(1.j*np.random.uniform(-np.pi, np.pi))
                squeezing_interactions.append(coupling)
                system.add_adag_adag_coupling(coupling, idx1, idx2)

        scattering_matrix, _ = calc_scattering_matrix(np.array([0.]), *system.return_system_parameters())
        conditions, _ =  setup_conditions(scattering_matrix[0], symbolic=False, conditions_using_adjugate=False)

        num_conditions = N_modes + (N_modes**2-N_modes)//2 + N_modes**2 + (N_modes**2-N_modes)//2 + N_modes**2
        np.testing.assert_array_almost_equal(conditions, np.zeros(num_conditions))

        modes = [Mode() for _ in range(N_modes)]
        operators = [m.a for m in modes] + [m.adag for m in modes]
        conditions_Bogoliubov, _ = setup_Bogoliubov_conditions(operators, sp.Matrix(scattering_matrix[0]), symbolic=False)
        num_dimensions = 2*N_modes
        np.testing.assert_array_almost_equal(np.asarray(conditions_Bogoliubov, dtype='complex'), np.zeros(num_dimensions+(num_dimensions**2-num_dimensions)//2))

class Numerical_solution_Tests(unittest.TestCase):
    def test_link_two_inputs_to_one_output_and_isolate_add_2_modes(self):
        conditions_expected = np.zeros(15)
        S_target = sp.sqrt(2)/2*sp.Matrix([[0,0,0],[0,0,0],[1,1,0]])
        free_parameters, S_extended = extend_matrix(S_target, 5, real_cs=True)
        conditions, info = setup_conditions(S_extended, S_upper_left=True)

        dyn_matrix = info['dynamical_matrix']

        initial_guess = np.zeros(len(free_parameters))

        solution, _ = find_numerical_solution_real_cs(
            conditions, free_parameters, initial_guess, method='least-squares'
        )

        conditions_evaluated = [float(condition.subs(solution)) for condition in conditions]

        np.testing.assert_array_almost_equal(conditions_evaluated, conditions_expected)

        S_extended_numpy = np.array(S_extended.subs(solution)).astype('float')

        np.testing.assert_array_almost_equal(S_extended_numpy[:3,:3], S_target)

        extracted_couplings = extract_coupling_rates(
            sp.N(dyn_matrix.subs(solution)),
            verbose=False,
        )

        test_system = Multimode_system(5)
        for idx1, idx2, coupling_rate in extracted_couplings:
            test_system.add_adag_a_coupling(coupling_rate, idx1, idx2)
        scattering, _ = calc_scattering_matrix(np.array([0]), *test_system.return_system_parameters())

        np.testing.assert_array_almost_equal(S_extended_numpy, scattering[0,:5,:5])

    def test_3_mode_unidirectional_amplifier(self):
        modes = [Mode() for _ in range(3)]
        operators = [modes[0].a, modes[1].a, modes[2].adag]
        S_target = sp.Matrix([[0,1],[10,0]])
        paras, S_extended = extend_matrix(S_target, 3, real_cs=True)
        conditions, _ = setup_conditions_subset(operators, S_extended)
        initial_guess = np.random.uniform(-0.5, 0.5, len(paras))
        solution, info = find_numerical_solution_real_cs(conditions, paras, initial_guess, method='lm')

        np.testing.assert_array_almost_equal(np.zeros(6), [c.subs(solution) for c in conditions])