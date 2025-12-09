import jax
import numpy as np
import jax.numpy as jnp
import sympy as sp
import scipy.optimize as sciopt
from tqdm import trange, tqdm
from itertools import product

import autolattice.constraints as msc
import autolattice.symbolic as sym
import autolattice.architecture as arch
from autolattice.jax_functions import adjugate

from autolattice.Custom_PSO import Custom_GlobalBestPSO
from pyswarms.single import LocalBestPSO
from autolattice.chain_functions import prepare_operators_chain

from autolattice.definitions import LEFT_TO_RIGHT, RIGHT_TO_LEFT

AUTODIFF_FORWARD = 'autodiff_forward'
AUTODIFF_REVERSE = 'autodiff_reverse'
DIFFERENCE_QUOTIENT = '2-point'



VAR_ABS = 'abs_variable'
VAR_PHASE = 'phase_variable'
VAR_INTRINSIC_LOSS = 'intrinsic_loss_variable'
VAR_EXTRINSIC_LOSS = 'extrinsic_loss_variable'
VAR_USER_DEFINED = 'user_defined'
VAR_EPSILON = 'epsilon_variable'

ZERO_LOSS_MODE = 'zero_loss_mode'
LOSSY_MODE = 'lossy_mode'

BIG_TO_SMALL = 'BIG_TO_SMALL'
SMALL_TO_BIG = 'SMALL_TO_BIG'

MODE_INPUT_NOISE = 'MODE_INPUT_NOISE'
MODE_OUTPUT_NOISE = 'MODE_INPUT_NOISE'

INIT_ABS_RANGE_DEFAULT = [-10., 10.]
INIT_INTRINSIC_LOSS_RANGE_DEFAULT = [-10., 10.]
INIT_EXTRINSIC_LOSS_RANGE_DEFAULT = [0., 20.]
# BOUNDS_ABS_DEFAULT = [-np.inf, np.inf]
BOUNDS_INTRINSIC_LOSS_DEFAULT = [-np.inf, np.inf]
DEFAULT_THRESHOLD = 1.e-6

EPSILON_MU1_DEFAULT = 1.e-8
# DEFAULT_PSO_PARAMETERS = {'num_particles': 50, 'max_iterations': 2000, 'options': {'c1': 0.5, 'c2': 0.3, 'w': 0.9}, 'bh_strategy': 'reflective'}
DEFAULT_PSO_PARAMETERS = {'num_particles': 50, 'max_iterations': 2000, 'options': {'c1': 0.2, 'c2': 0.9, 'w': 0.9}, 'bh_strategy': 'nearest'}

def calc_scattering_matrix_from_coupling_matrix(coupling_matrix, kappa_int_matrix, kappa_ext_matrix, omega=0.):
    num_modes = coupling_matrix.shape[0]
    identity = jnp.eye(num_modes)

    sqrt_kappa_ext_matrix = jnp.sqrt(kappa_ext_matrix)
    
    # if all matrices (except kappa_ext) would be rescaled
    # scattering_matrix = identity + jnp.linalg.inv(-1.j*coupling_matrix_dimensionless + 1.j*omega*jnp.linalg.inv(kappa_ext_matrix) - (identity + kappa_int_matrix_dimensionless)/2. )

    scattering_matrix = identity + sqrt_kappa_ext_matrix@jnp.linalg.inv(-1.j*coupling_matrix + 1.j*omega*identity - (kappa_ext_matrix+kappa_int_matrix)/2.)@sqrt_kappa_ext_matrix
    return scattering_matrix

def create_solutions_dict(variables, values):
    return {variable.name: value for variable, value in zip(variables, values)}

class Architecture_Optimizer():
    def __init__(
            self,
            modes_per_unit_cell,
            S_target=None, 
            S_target_elements=None,
            mode_types='no_squeezing',
            gradient_method=AUTODIFF_REVERSE,
            kwargs_optimization={},
            pso_parameters={},
            S_target_free_symbols_init_range=None,
            make_initial_test=False,
            phase_constraints_for_squeezing=False,
            port_intrinsic_losses=False,
            kappas_free_parameters=False,
            method=None,
            logfile='log',
            enforced_constraints=[]):
        '''
        S_target: list of elements which the scattering matrix has to match; if an element has a free variable, this variable will be optimised during the optimisation
        S_target_elements: specifies which elements within the scattering matrix have to match the input provided in S_target
            example input: 
            S_target = [0, x, 1] # x is a real sympy variable
            S_target_elements = [(0,1), (2,3), (3,-1)] 
            # the element (0,1) has to equal 0, the element (2,3) has to equal x, which is a free variable, and the element (3,-1) has to match 1
        kappas_free_parameters: list of kappas that can be chosen freely, length has to equal modes_per_unit_cell,
            elements with False correspond to kappas set to 1., elements with True are free parameters left for optimisation,
            if kappas_free_parameters is set to False, all elements are set to False (default option)
            if kappas_free_parameters is set to True, all elements except the first are set to True
        '''

        self.kwargs_optimization = {
            'num_tests': 10,
            'verbosity': 0,
            'kwargs_initialisation': {},
            'threshold_accept_solution': DEFAULT_THRESHOLD, 
            'interrupt_if_successful': True,
            'kwargs_bounds': {'bounds_extrinsic_loss': [0., jnp.inf]},
            'log_full_swarm_history': False
        }
        self.kwargs_optimization.update(kwargs_optimization)
        self.pso_parameters = DEFAULT_PSO_PARAMETERS.copy()
        self.pso_parameters.update(pso_parameters)

        self.modes_per_unit_cell = modes_per_unit_cell

        if logfile is not None:
            self.logfile = open(logfile, 'w')
        else:
            self.logfile = None

        if S_target is None:
            self.S_target = sp.zeros(0)
        else:
            self.S_target = sp.Matrix(S_target)

        self.S_target_elements = S_target_elements
        if S_target_elements is None:
            self.S_target_mask = (np.zeros(0, dtype='int'), np.zeros(0, dtype='int'))
        else:
            self.S_target_mask = (jnp.array(S_target_elements)[:,0], jnp.array(S_target_elements)[:,1])

        if port_intrinsic_losses is True:
            self.port_intrinsic_losses = [True for _ in range(self.modes_per_unit_cell)]
        elif port_intrinsic_losses is False:
            self.port_intrinsic_losses = [False for _ in range(self.modes_per_unit_cell)]
        else:
            self.port_intrinsic_losses = port_intrinsic_losses

        if kappas_free_parameters is True:
            kappas_free_parameters = [False] + [True for _ in range(self.modes_per_unit_cell - 1)]
        elif kappas_free_parameters is False:
            kappas_free_parameters = [False for _ in range(self.modes_per_unit_cell)]
        else:
            kappas_free_parameters = list(kappas_free_parameters)
        self.kappas_free_parameters = kappas_free_parameters
        if len(kappas_free_parameters) != self.modes_per_unit_cell:
            raise ValueError('length of kappas_free_parameters has to equal modes_per_unit_cell')

        self.__prepare_mode_types__(mode_types)  
        self.phase_constraints_for_squeezing = phase_constraints_for_squeezing
        self.enforced_constraints = enforced_constraints
        self.__setup_all_constraints__()

        self.Deltas = []
        self.gabs = [] 
        self.gphases = [] 


        # initialize everything to estimate gain rate and prefactor for infinite chain_length limit
        # __initialize_coupling_matrix__ called with append=True will also initialize all coupling matrix parameters
        self.coupling_matrix_infinite = self.__give_coupling_matrix__(chain_length=3, append=True)
        # initialize all parameters excluding those of the coupling matrix
        self.__initialize_parameters__(S_target_free_symbols_init_range)

        self.kappa_int_matrix_infinite, self.kappa_ext_matrix_infinite = self.__give_kappa_matrices__(chain_length=3)
        # jaxify matrices
        self.coupling_matrix_infinite_jax = self.__jaxify_function__(self.coupling_matrix_infinite)
        self.kappa_int_matrix_infinite_jax = self.__jaxify_function__(self.kappa_int_matrix_infinite)
        self.kappa_ext_matrix_infinite_jax = self.__jaxify_function__(self.kappa_ext_matrix_infinite)

        self.conditions_func_unjaxed = self.__initialize_all_conditions_func__()
        self.conditions_func = jax.jit(self.conditions_func_unjaxed)
        self.conditions_func_swarm = jax.jit(jax.vmap(self.conditions_func_unjaxed))
        
        self.gradient_method = gradient_method
        if gradient_method == AUTODIFF_FORWARD:
            self.jacobian = jax.jit(jax.jacfwd(self.conditions_func, has_aux=True))
        elif gradient_method == AUTODIFF_REVERSE:
            self.jacobian = jax.jit(jax.jacrev(self.conditions_func, has_aux=True))
        elif gradient_method == DIFFERENCE_QUOTIENT:
            self.jacobian = None
        else:
            raise NotImplementedError()
        
        self.valid_combinations = []
        self.invalid_combinations = []
        self.tested_complexities = []
        
        # # make run without additional conditions
        if make_initial_test:
            success, _, _ = self.repeated_optimization(conditions=[])
            if not success:
                print('unconditioned system cannot be solved, interrupting')

    def __provide_function_for_exact_scattering_calculation__(self, chain_length):
        coupling_matrix = self.__give_coupling_matrix__(chain_length)
        kappa_int_matrix, kappa_ext_matrix = self.__give_kappa_matrices__(chain_length)

        coupling_matrix_jax = self.__jaxify_function__(coupling_matrix)
        kappa_int_matrix_jax = self.__jaxify_function__(kappa_int_matrix)
        kappa_ext_matrix_jax = self.__jaxify_function__(kappa_ext_matrix)

        def calc_scattering_matrix_for_parameters(omega, input_array):
            coupling_matrix = coupling_matrix_jax(*input_array)
            kappa_int_matrix = kappa_int_matrix_jax(*input_array)
            kappa_ext_matrix = kappa_ext_matrix_jax(*input_array)
            scattering_matrix = calc_scattering_matrix_from_coupling_matrix(coupling_matrix, kappa_int_matrix, kappa_ext_matrix, omega=omega)
            return scattering_matrix

        info_dict = {
            'coupling_matrix': coupling_matrix,
            'kappa_int_matrix': kappa_int_matrix,
            'kappa_ext_matrix': kappa_ext_matrix,
            'coupling_matrix_jax': coupling_matrix_jax,
            'kappa_int_matrix_jax': kappa_int_matrix_jax,
            'kappa_ext_matrix_jax': kappa_ext_matrix_jax
        }

        return calc_scattering_matrix_for_parameters, info_dict
    
    def __provide_function_for_dynamical_matrix__(self, chain_length, OBC=True):
        coupling_matrix = self.__give_coupling_matrix__(chain_length, OBC=OBC)
        kappa_int_matrix, kappa_ext_matrix = self.__give_kappa_matrices__(chain_length)

        coupling_matrix_jax = self.__jaxify_function__(coupling_matrix)
        kappa_int_matrix_jax = self.__jaxify_function__(kappa_int_matrix)
        kappa_ext_matrix_jax = self.__jaxify_function__(kappa_ext_matrix)

        def calc_dynamical_matrix(input_array):
            coupling_matrix = coupling_matrix_jax(*input_array)
            kappa_int_matrix = kappa_int_matrix_jax(*input_array)
            kappa_ext_matrix = kappa_ext_matrix_jax(*input_array)
            kappa_ext_matrix_sqrt = jnp.sqrt(kappa_ext_matrix)

            # coupling_matrix = kappa_ext_matrix_sqrt@coupling_matrix@kappa_ext_matrix_sqrt
            # kappa_int_matrix = kappa_int_matrix@kappa_ext_matrix

            return -1.j*coupling_matrix - (kappa_int_matrix + kappa_ext_matrix)/2.
        
        return calc_dynamical_matrix

    def __setup_all_constraints__(self):
        self.all_possible_constraints = []

        for idx in range(self.modes_per_unit_cell):
            self.all_possible_constraints.append(msc.Constraint_coupling_zero(idx, idx, self.modes_per_unit_cell))
        for idx2 in range(2*self.modes_per_unit_cell):
            for idx1 in range(idx2):
                if type(self.operators_chain_two_unit_cells[idx1]) is type(self.operators_chain_two_unit_cells[idx2]):
                    self.all_possible_constraints.append(msc.Constraint_coupling_zero(idx1, idx2, self.modes_per_unit_cell))
                    self.all_possible_constraints.append(msc.Constraint_coupling_phase_zero(idx1, idx2, self.modes_per_unit_cell))
                else:
                    self.all_possible_constraints.append(msc.Constraint_coupling_zero(idx1, idx2, self.modes_per_unit_cell))
                    if self.phase_constraints_for_squeezing:
                        self.all_possible_constraints.append(msc.Constraint_coupling_phase_zero(idx1, idx2, self.modes_per_unit_cell))

    def give_mus(self, input_array):
        if self.mode_types[-1] != self.mode_types[0]:
            raise NotImplementedError("not completely clear how you would have to define the mu's")

        coupling_matrix = self.coupling_matrix_infinite_jax(*input_array)
        kappa_int_matrix = self.kappa_int_matrix_infinite_jax(*input_array)
        kappa_ext_matrix = self.kappa_ext_matrix_infinite_jax(*input_array)

        dynamical_matrix = -1.j*coupling_matrix - (kappa_ext_matrix + kappa_int_matrix)/2. 

        mu0 = dynamical_matrix[:self.modes_per_unit_cell,:self.modes_per_unit_cell]
        mu1 = dynamical_matrix[:self.modes_per_unit_cell,self.modes_per_unit_cell:2*self.modes_per_unit_cell]
        mu_1 = dynamical_matrix[self.modes_per_unit_cell:2*self.modes_per_unit_cell,:self.modes_per_unit_cell]

        return mu_1, mu0, mu1

    def create_transfer_matrix(self, omega, mu_1, mu0, mu1, epsilon_mu1=EPSILON_MU1_DEFAULT):
        mu1_inv = self.invert_mu1(mu1, epsilon_mu1=epsilon_mu1)
        identity = jnp.eye(self.modes_per_unit_cell)
        zero_matrix = jnp.zeros([self.modes_per_unit_cell,self.modes_per_unit_cell])

        transfer_matrix = jnp.vstack((jnp.hstack((-mu1_inv@(mu0 + 1.j*omega*identity), -mu1_inv@(mu_1+identity*epsilon_mu1))), jnp.hstack((identity, zero_matrix))))

        return transfer_matrix

    def give_projector(self, direction):
        zero_matrix = jnp.zeros([self.modes_per_unit_cell,self.modes_per_unit_cell])
        identity = jnp.eye(self.modes_per_unit_cell)

        if direction == LEFT_TO_RIGHT:
            projector = jnp.hstack((zero_matrix, identity))
        elif direction == RIGHT_TO_LEFT:
            projector = jnp.hstack((identity, zero_matrix))
        else:
            raise NotImplementedError()
        
        return projector

    def calc_gain_exact(self, omega, input_array, chain_length, unit_cell_idx=None, epsilon_mu1=EPSILON_MU1_DEFAULT, direction=LEFT_TO_RIGHT):
        if unit_cell_idx is None:
            unit_cell_idx = chain_length

        mu_1, mu0, mu1 = self.give_mus(input_array)
        transfer_matrix = self.create_transfer_matrix(omega, mu_1, mu0, mu1, epsilon_mu1=epsilon_mu1)
        
        kappa_ext_matrix_unit_cell = self.kappa_ext_matrix_infinite_jax(*input_array)[:self.modes_per_unit_cell,:self.modes_per_unit_cell]

        projector = self.give_projector(direction=direction)

        if direction == RIGHT_TO_LEFT:
            product_denom = jnp.linalg.matrix_power(transfer_matrix, chain_length)    
            product_nom = jnp.linalg.matrix_power(transfer_matrix, chain_length-unit_cell_idx)
            mu1_inv = self.invert_mu1(mu1, epsilon_mu1=epsilon_mu1)
        elif direction == LEFT_TO_RIGHT:
            product_denom = jnp.linalg.matrix_power(transfer_matrix, -chain_length)
            product_nom = jnp.linalg.matrix_power(transfer_matrix, -(unit_cell_idx-1))
            mu1_inv = self.invert_mu1(mu_1, epsilon_mu1=epsilon_mu1)
        else:
            raise NotImplementedError()

        scattering = - jnp.sqrt(kappa_ext_matrix_unit_cell) @ jnp.linalg.inv(projector@product_denom@(projector.T)) @ (projector@product_nom@(projector.T)) @ mu1_inv @ jnp.sqrt(kappa_ext_matrix_unit_cell)

        if (direction is LEFT_TO_RIGHT and unit_cell_idx == chain_length) or (direction is RIGHT_TO_LEFT and unit_cell_idx == 1):
            scattering += jnp.eye(self.modes_per_unit_cell)

        return scattering
    
    def invert_mu1(self, mu1, epsilon_mu1=EPSILON_MU1_DEFAULT):
        identity = jnp.eye(self.modes_per_unit_cell)
        return jnp.linalg.inv(mu1 + identity * epsilon_mu1)

    def calc_eigensystem_from_parameters(self, omega, input_array, sort_order=SMALL_TO_BIG, epsilon_mu1=EPSILON_MU1_DEFAULT):
        mu_1, mu0, mu1 = self.give_mus(input_array)
        transfer_matrix = self.create_transfer_matrix(omega, mu_1, mu0, mu1, epsilon_mu1=epsilon_mu1)

        return self.calc_eigensystem(transfer_matrix, sort_order)

    def calc_eigensystem(self, transfer_matrix, sort_order=SMALL_TO_BIG):
        # ith right eigenvector is right_eigvectors[:,i]
        # ith left eigenvector is left_eigvectors[i,:]
        if sort_order == BIG_TO_SMALL:
            eigvals, right_eigvectors = jnp.linalg.eig(transfer_matrix)
            idx_sort = jnp.flip(jnp.argsort(jnp.abs(eigvals))) #sorts from largest to smallest eigenvalue
            eigvals = eigvals[idx_sort]
            right_eigvectors = right_eigvectors[:,idx_sort]
            left_eigvectors = jnp.linalg.inv(right_eigvectors)
        elif sort_order == SMALL_TO_BIG:
            eigvals, right_eigvectors = jnp.linalg.eig(transfer_matrix)
            idx_sort = jnp.argsort(jnp.abs(eigvals)) #sorts from smallest to largest eigenvalue
            eigvals = eigvals[idx_sort]
            right_eigvectors = right_eigvectors[:,idx_sort]
            left_eigvectors = jnp.linalg.inv(right_eigvectors)

        outer_products = []  #submatrices of the rank-1 matrices
        for eigval_idx in range(2*self.modes_per_unit_cell):
            outer_product = jnp.outer(right_eigvectors[:,eigval_idx], left_eigvectors[eigval_idx,:])
            outer_products.append(outer_product)
        
        # outer_products = None

        return eigvals, right_eigvectors, left_eigvectors, outer_products

    def calc_matrix_inversion_prefactor(self, right_eigvectors, left_eigvectors, projector):

        # ith right eigenvector is right_eigvectors[:,i]
        # ith left eigenvector is left_eigvectors[i,:]

        # select the first M eigenvectors
        right_eigvectors_first_M = right_eigvectors[:,:self.modes_per_unit_cell]
        left_eigvectors_first_M = left_eigvectors[:self.modes_per_unit_cell,:]

        # project them using the projector
        U = projector@right_eigvectors_first_M
        V = left_eigvectors_first_M@(projector.T) # note, that V is actually V^T in the paper

        diag_matrix = jnp.zeros([self.modes_per_unit_cell,self.modes_per_unit_cell])
        diag_matrix = diag_matrix.at[self.modes_per_unit_cell - 1, self.modes_per_unit_cell - 1].set(1)

        approximated_inverse_prefactor = jnp.linalg.inv(V) @ diag_matrix @ jnp.linalg.inv(U)
        approximated_inverse_prefactor_bulk = jnp.linalg.inv(V) @ diag_matrix @ V

        # U = projector @ right_eigvectors @ projector.T
        # V = projector @ left_eigvectors @ projector.T

        # diag_matrix = jnp.zeros([self.modes_per_unit_cell,self.modes_per_unit_cell])
        # diag_matrix = diag_matrix.at[self.modes_per_unit_cell - 1, self.modes_per_unit_cell - 1].set(1)

        # approximated_inverse_prefactor = jnp.linalg.inv(V) @ diag_matrix @ jnp.linalg.inv(U)

        # if self.modes_per_unit_cell == 1:
        #     approximated_inverse_prefactor = jnp.linalg.inv(projector@Xs[0]@(projector.T))
        # # elif self.modes_per_unit_cell == 2:
        # #     adj_X0 = adjugate(projector@Xs[0]@(projector.T))
        # #     approximated_inverse_prefactor = adj_X0 / jnp.trace(adj_X0 @ (projector@Xs[1]@(projector.T)))
        # # else:
        # #     raise NotImplementedError()
        # else:
        #     Xs_projected = [projector@element@(projector.T) for element in Xs]
        #     sum_for_adjugate = jnp.zeros([self.modes_per_unit_cell,self.modes_per_unit_cell])
        #     for j in range(self.modes_per_unit_cell-1):
        #         sum_for_adjugate += Xs_projected[j]
        #     adjugate_term = adjugate(sum_for_adjugate)
        #     approximated_inverse_prefactor = adjugate_term/jnp.trace(adjugate_term@Xs_projected[self.modes_per_unit_cell-1])
        
        return approximated_inverse_prefactor, approximated_inverse_prefactor_bulk
    
    def calc_gain_scaling_bulk(self, omega, input_array, epsilon_mu1=EPSILON_MU1_DEFAULT, return_info=False, direction=LEFT_TO_RIGHT):
        # raise NotImplementedError()
        prefactor_at_boundary, scaling_rate, info = self.calc_gain_scaling(omega, input_array, epsilon_mu1=epsilon_mu1, return_info=True, direction=direction)
        # outer_products = info['outer_products']
        # approximated_inverse_prefactor = info['approximated_inverse_prefactor']
        # kappa_ext_matrix_unit_cell = info['kappa_ext_matrix_unit_cell']
        # mu1_inv = info['mu1_inv']
        # projector = info['projector']

        # X_M_projected = projector@outer_products[self.modes_per_unit_cell-1]@(projector.T)

        # approximated_inverse_prefactor_bulk = approximated_inverse_prefactor@X_M_projected
        # prefactor_bulk = - jnp.sqrt(kappa_ext_matrix_unit_cell) @ approximated_inverse_prefactor_bulk @ mu1_inv @ jnp.sqrt(kappa_ext_matrix_unit_cell)
        prefactor_bulk = info['prefactor_bulk']

        if not return_info:
            return prefactor_bulk, scaling_rate
        else:
            return prefactor_bulk, scaling_rate, info
        
    def calc_gain_scaling_prefactor_bulk(self, omega, input_array, epsilon_mu1=EPSILON_MU1_DEFAULT, direction=LEFT_TO_RIGHT):
        prefactor_bulk, gain_rate = self.calc_gain_scaling_bulk(omega, input_array, epsilon_mu1=epsilon_mu1, direction=direction)
        return prefactor_bulk

    def calc_gain_scaling(self, omega, input_array, epsilon_mu1=EPSILON_MU1_DEFAULT, return_info=False, direction=LEFT_TO_RIGHT):
        mu_1, mu0, mu1 = self.give_mus(input_array)
        transfer_matrix = self.create_transfer_matrix(omega, mu_1, mu0, mu1, epsilon_mu1=epsilon_mu1)
        
        kappa_ext_matrix_unit_cell = self.kappa_ext_matrix_infinite_jax(*input_array)[:self.modes_per_unit_cell,:self.modes_per_unit_cell]

        if direction == LEFT_TO_RIGHT:
            sort_order = SMALL_TO_BIG
            mu1_inv = self.invert_mu1(mu_1, epsilon_mu1=epsilon_mu1)
        elif direction == RIGHT_TO_LEFT:
            sort_order = BIG_TO_SMALL
            mu1_inv = self.invert_mu1(mu1, epsilon_mu1=epsilon_mu1)
        else:
            raise NotImplementedError()

        projector = self.give_projector(direction)
        eigvals, right_eigvectors, left_eigvectors, outer_products = self.calc_eigensystem(transfer_matrix, sort_order)
        approximated_inverse_prefactor, approximated_inverse_prefactor_bulk = self.calc_matrix_inversion_prefactor(right_eigvectors, left_eigvectors, projector)
        prefactor = - jnp.sqrt(kappa_ext_matrix_unit_cell) @ approximated_inverse_prefactor @ mu1_inv @ jnp.sqrt(kappa_ext_matrix_unit_cell)
        prefactor_bulk = - jnp.sqrt(kappa_ext_matrix_unit_cell) @ approximated_inverse_prefactor_bulk @ mu1_inv @ jnp.sqrt(kappa_ext_matrix_unit_cell)
        dominant_eigval = eigvals[self.modes_per_unit_cell-1]

        if direction == LEFT_TO_RIGHT:
            scaling_rate = dominant_eigval
        elif direction == RIGHT_TO_LEFT:
            scaling_rate = 1./dominant_eigval

        if not return_info:
            return prefactor, scaling_rate
        else:
            info = {
                'outer_products': outer_products,
                'approximated_inverse_prefactor': approximated_inverse_prefactor,
                'approximated_inverse_prefactor_bulk': approximated_inverse_prefactor_bulk,
                'prefactor_bulk': prefactor_bulk,
                'eigvals': eigvals,
                'right_eigvectors': right_eigvectors,
                'left_eigvectors': left_eigvectors,
                'mu1_inv': mu1_inv,
                'kappa_ext_matrix_unit_cell': kappa_ext_matrix_unit_cell,
                'projector': projector
            }
            return prefactor, scaling_rate, info
    
    # def calc_noise(self, omega, input_array)

    def calc_gain_scaling_rate(self, omega, input_array, direction=LEFT_TO_RIGHT, epsilon_mu1=EPSILON_MU1_DEFAULT, idx_eigval=None):
        # prefactor, gain_rate = self.calc_gain_scaling(omega, input_array, direction=direction)
        # return gain_rate
        eigvals = self.calc_eigvals_transfer_matrix(omega, input_array, direction=direction, epsilon_mu1=epsilon_mu1)

        if idx_eigval is None: #if None select the leading order contribution
            selected_eigval = eigvals[self.modes_per_unit_cell-1]
        else:
            selected_eigval = eigvals[idx_eigval]

        if direction == LEFT_TO_RIGHT:
            return jnp.abs(selected_eigval)
        elif direction == RIGHT_TO_LEFT:
            return 1./jnp.abs(selected_eigval)
    
    def calc_eigvals_transfer_matrix(self, omega, input_array, direction=LEFT_TO_RIGHT, epsilon_mu1=EPSILON_MU1_DEFAULT):
        if direction == LEFT_TO_RIGHT:
            sort_order = SMALL_TO_BIG
        elif direction == RIGHT_TO_LEFT:
            sort_order = BIG_TO_SMALL
        else:
            raise NotImplementedError()

        mu_1, mu0, mu1 = self.give_mus(input_array)
        transfer_matrix = self.create_transfer_matrix(omega, mu_1, mu0, mu1, epsilon_mu1=epsilon_mu1)

        eigvals = jnp.linalg.eigvals(transfer_matrix)
        if sort_order == BIG_TO_SMALL:
            idx_sort = jnp.flip(jnp.argsort(jnp.abs(eigvals))) #sorts from largest to smallest eigenvalue
        elif sort_order == SMALL_TO_BIG:
            idx_sort = jnp.argsort(jnp.abs(eigvals)) #sorts from smallest to largest eigenvalue
        eigvals = eigvals[idx_sort]

        return eigvals

    def calc_difference_between_eigvals(self, omega, input_array, epsilon_mu1=EPSILON_MU1_DEFAULT):
        eigvals = self.calc_eigvals_transfer_matrix(omega, input_array, epsilon_mu1=epsilon_mu1)
        eigvals_abs = jnp.abs(eigvals)

        return \
            jnp.array([
                jnp.abs(eigvals_abs[self.modes_per_unit_cell-1]-eigvals_abs[self.modes_per_unit_cell]),
                jnp.abs(eigvals_abs[self.modes_per_unit_cell-1]-eigvals_abs[self.modes_per_unit_cell-2])]), \
            jnp.array([
                jnp.abs(eigvals[self.modes_per_unit_cell-1]-eigvals[self.modes_per_unit_cell]),
                jnp.abs(eigvals[self.modes_per_unit_cell-1]-eigvals[self.modes_per_unit_cell-2])
        ])
        
    def calc_gain_scaling_prefactor(self, omega, input_array, epsilon_mu1=EPSILON_MU1_DEFAULT, direction=LEFT_TO_RIGHT):
        prefactor, gain_rate = self.calc_gain_scaling(omega, input_array, epsilon_mu1=epsilon_mu1, direction=direction)
        return prefactor
    
    def calc_gain_approximate(self, omega, input_array, chain_length, epsilon_mu1=EPSILON_MU1_DEFAULT, direction=LEFT_TO_RIGHT):
        prefactor, scaling_rate = self.calc_gain_scaling(omega, input_array, epsilon_mu1=epsilon_mu1, direction=direction)
        return prefactor*scaling_rate**chain_length

    def calc_gain_approximation_intermediate_mode(self, omega, input_array, chain_length, unit_cell_idx, epsilon_mu1=EPSILON_MU1_DEFAULT, direction=LEFT_TO_RIGHT):
        raise NotImplementedError()
        mu_1, mu0, mu1 = self.give_mus(input_array)
        transfer_matrix = self.create_transfer_matrix(omega, mu_1, mu0, mu1, epsilon_mu1=epsilon_mu1)
        
        kappa_ext_matrix_unit_cell = self.kappa_ext_matrix_infinite_jax(*input_array)[:self.modes_per_unit_cell,:self.modes_per_unit_cell]
        sqrt_kappa_ext = jnp.sqrt(kappa_ext_matrix_unit_cell)

        if direction == LEFT_TO_RIGHT:
            sort_order = SMALL_TO_BIG
            mu1_inv = self.invert_mu1(mu_1, epsilon_mu1=epsilon_mu1)
        elif direction == RIGHT_TO_LEFT:
            sort_order = BIG_TO_SMALL
            mu1_inv = self.invert_mu1(mu1, epsilon_mu1=epsilon_mu1)
        else:
            raise NotImplementedError()

        projector = self.give_projector(direction)
        eigvals, right_eigvectors, left_eigvectors, outer_products = self.calc_eigensystem(transfer_matrix, sort_order)
        Xs = [projector@u_v@(projector.T) for u_v in outer_products]
        approximated_inverse_prefactor = self.calc_matrix_inversion_prefactor(outer_products, projector)

        dominant_eigval = eigvals[self.modes_per_unit_cell-1]

        if direction == LEFT_TO_RIGHT:
            scaling_rate = dominant_eigval
        elif direction == RIGHT_TO_LEFT:
            scaling_rate = 1./dominant_eigval


        if (unit_cell_idx == chain_length and direction == RIGHT_TO_LEFT) or (unit_cell_idx == 1 and direction == LEFT_TO_RIGHT):
            prefactor = - sqrt_kappa_ext @ approximated_inverse_prefactor @ mu1_inv @ sqrt_kappa_ext
            return prefactor * scaling_rate**chain_length
        else:
            K = self.modes_per_unit_cell + 1
            correction_factor = jnp.zeros([self.modes_per_unit_cell,self.modes_per_unit_cell])
            for n in range(K):
                X_idx = self.modes_per_unit_cell - 1 + n
                if direction == LEFT_TO_RIGHT:
                    exponent = 1-unit_cell_idx
                elif direction == RIGHT_TO_LEFT:
                    exponent = chain_length-unit_cell_idx
                correction_factor += Xs[X_idx] * eigvals[X_idx] ** exponent
            return -sqrt_kappa_ext @ approximated_inverse_prefactor @ correction_factor @ mu1_inv @ sqrt_kappa_ext * scaling_rate**chain_length

    def calc_reverse_gain_approximate_intermediate(self, omega, input_array, chain_length, unit_cell_idx, epsilon_mu1=EPSILON_MU1_DEFAULT, direction=RIGHT_TO_LEFT):
        raise NotImplementedError()
        mu_1, mu0, mu1 = self.give_mus(input_array)
        transfer_matrix = self.create_transfer_matrix(omega, mu_1, mu0, mu1, epsilon_mu1=epsilon_mu1)
        
        kappa_ext_matrix_unit_cell = self.kappa_ext_matrix_infinite_jax(*input_array)[:self.modes_per_unit_cell,:self.modes_per_unit_cell]
        sqrt_kappa_ext = jnp.sqrt(kappa_ext_matrix_unit_cell)

        if direction == LEFT_TO_RIGHT:
            sort_order = SMALL_TO_BIG
            mu1_inv = self.invert_mu1(mu_1, epsilon_mu1=epsilon_mu1)
        elif direction == RIGHT_TO_LEFT:
            sort_order = BIG_TO_SMALL
            mu1_inv = self.invert_mu1(mu1, epsilon_mu1=epsilon_mu1)
        else:
            raise NotImplementedError()

        eigvals, right_eigvectors, left_eigvectors, outer_products = self.calc_eigensystem(transfer_matrix, sort_order)
        if self.modes_per_unit_cell != 2 or direction != RIGHT_TO_LEFT:
            raise NotImplementedError()
        
        projector = self.give_projector(direction)
        Xs = [projector@u_v@(projector.T) for u_v in outer_products]

        scattering_matrix_element = jnp.zeros([self.modes_per_unit_cell, self.modes_per_unit_cell], dtype='complex')
        if unit_cell_idx == 1:
            scattering_matrix_element += jnp.eye(self.modes_per_unit_cell)

        factor = adjugate(Xs[1])@Xs[0] * eigvals[0] ** -unit_cell_idx + adjugate(Xs[0])@Xs[1] * eigvals[1] ** -unit_cell_idx
        prefactor = 1/jnp.trace(adjugate(Xs[0])@Xs[1])
        scattering_matrix_element -= prefactor * sqrt_kappa_ext @ factor @ mu1_inv @ sqrt_kappa_ext
        return scattering_matrix_element

    def calc_input_noise_on_resonance(self, input_array, input_idx, amplification_direction=LEFT_TO_RIGHT, bath_occupations=None, epsilon_mu1=EPSILON_MU1_DEFAULT):
        raise NotImplementedError()
        if bath_occupations is None:
            bath_occupations = jnp.zeros(self.modes_per_unit_cell)

        if amplification_direction is not LEFT_TO_RIGHT:
            raise NotImplementedError
        direction = RIGHT_TO_LEFT

        mu_1, mu0, mu1 = self.give_mus(input_array)
        transfer_matrix = self.create_transfer_matrix(0., mu_1, mu0, mu1, epsilon_mu1=epsilon_mu1)

        if direction == LEFT_TO_RIGHT:
            sort_order = SMALL_TO_BIG
            mu1_inv = self.invert_mu1(mu_1, epsilon_mu1=epsilon_mu1)
        elif direction == RIGHT_TO_LEFT:
            sort_order = BIG_TO_SMALL
            mu1_inv = self.invert_mu1(mu1, epsilon_mu1=epsilon_mu1)
        else:
            raise NotImplementedError()
        
        kappa_ext_matrix_unit_cell = self.kappa_ext_matrix_infinite_jax(*input_array)[:self.modes_per_unit_cell,:self.modes_per_unit_cell]
        sqrt_kappa_ext = jnp.sqrt(kappa_ext_matrix_unit_cell)

        eigvals, right_eigvectors, left_eigvectors, outer_products = self.calc_eigensystem(transfer_matrix, sort_order)
        if self.modes_per_unit_cell != 2 or direction != RIGHT_TO_LEFT:
            raise NotImplementedError()
        
        projector = self.give_projector(direction)
        Xs = [projector@u_v@(projector.T) for u_v in outer_products]

        Akm = - 1/jnp.trace(adjugate(Xs[0])@Xs[1]) * sqrt_kappa_ext @ adjugate(Xs[1])@Xs[0] @ mu1_inv @ sqrt_kappa_ext
        Bkm = - 1/jnp.trace(adjugate(Xs[0])@Xs[1]) * sqrt_kappa_ext @ adjugate(Xs[0])@Xs[1] @ mu1_inv @ sqrt_kappa_ext

        help_matrix_Ck = jnp.eye(self.modes_per_unit_cell) + Akm * eigvals[0]**(-1) + Bkm * eigvals[1]**(-1)
        Ck = 0.5 * jnp.sum(jnp.abs(help_matrix_Ck[input_idx])**2 * (2*bath_occupations + 1))

        Dk = 0.5 * jnp.sum(jnp.abs(Akm[input_idx])**2 * (2*bath_occupations + 1))
        Ek = 0.5 * jnp.sum(jnp.abs(Bkm[input_idx])**2 * (2*bath_occupations + 1))
        Fk = 0.5 * jnp.sum(Akm[input_idx] * jnp.conjugate(Bkm[input_idx]) * (2*bath_occupations + 1))

        def func(r):
            return r**2/(1-r)
        
        return Ck + Dk * func(jnp.abs(eigvals[0])**-2) + Ek * func(jnp.abs(eigvals[1])**-2) + 2*jnp.real(Fk * func(1/(eigvals[0]*jnp.conjugate(eigvals[1]))))

    def calc_reflection_at_input(self, omega, input_array, input_idx, amplification_direction=LEFT_TO_RIGHT, epsilon_mu1=EPSILON_MU1_DEFAULT):
        raise NotImplementedError()
        if amplification_direction is not LEFT_TO_RIGHT:
            raise NotImplementedError
        direction = RIGHT_TO_LEFT

        mu_1, mu0, mu1 = self.give_mus(input_array)
        transfer_matrix = self.create_transfer_matrix(omega, mu_1, mu0, mu1, epsilon_mu1=epsilon_mu1)

        if direction == LEFT_TO_RIGHT:
            sort_order = SMALL_TO_BIG
            mu1_inv = self.invert_mu1(mu_1, epsilon_mu1=epsilon_mu1)
        elif direction == RIGHT_TO_LEFT:
            sort_order = BIG_TO_SMALL
            mu1_inv = self.invert_mu1(mu1, epsilon_mu1=epsilon_mu1)
        else:
            raise NotImplementedError()
        
        kappa_ext_matrix_unit_cell = self.kappa_ext_matrix_infinite_jax(*input_array)[:self.modes_per_unit_cell,:self.modes_per_unit_cell]
        sqrt_kappa_ext = jnp.sqrt(kappa_ext_matrix_unit_cell)

        eigvals, right_eigvectors, left_eigvectors, outer_products = self.calc_eigensystem(transfer_matrix, sort_order)
        if self.modes_per_unit_cell != 2 or direction != RIGHT_TO_LEFT:
            raise NotImplementedError()
        
        projector = self.give_projector(direction)
        Xs = [projector@u_v@(projector.T) for u_v in outer_products]


        prefactor = 1/jnp.trace(adjugate(Xs[0])@Xs[1])
        term1 = 1/eigvals[0] * adjugate(Xs[1])@Xs[0]
        term2 = 1/eigvals[1] * adjugate(Xs[0])@Xs[1]
        S11 = jnp.eye(self.modes_per_unit_cell) - prefactor * sqrt_kappa_ext @ (term1 + term2) @ mu1_inv @ sqrt_kappa_ext
        return S11[input_idx, input_idx]

    def calc_output_noise_on_resonance(self, input_array, input_idx, output_idx, amplification_direction=LEFT_TO_RIGHT, bath_occupations=None, epsilon_mu1=EPSILON_MU1_DEFAULT):
        raise NotImplementedError()
        if bath_occupations is None:
            bath_occupations = jnp.zeros(self.modes_per_unit_cell)

        # if amplification_direction == LEFT_TO_RIGHT:
        #     raise NotImplementedError()
        
        direction = amplification_direction

        mu_1, mu0, mu1 = self.give_mus(input_array)
        transfer_matrix = self.create_transfer_matrix(0., mu_1, mu0, mu1, epsilon_mu1=epsilon_mu1)
        kappa_ext_matrix_unit_cell = self.kappa_ext_matrix_infinite_jax(*input_array)[:self.modes_per_unit_cell,:self.modes_per_unit_cell]
        sqrt_kappa_ext = jnp.sqrt(kappa_ext_matrix_unit_cell)

        if direction == LEFT_TO_RIGHT:
            sort_order = SMALL_TO_BIG
            mu1_inv = self.invert_mu1(mu_1, epsilon_mu1=epsilon_mu1)
        elif direction == RIGHT_TO_LEFT:
            sort_order = BIG_TO_SMALL
            mu1_inv = self.invert_mu1(mu1, epsilon_mu1=epsilon_mu1)
        else:
            raise NotImplementedError()

        eigvals, right_eigvectors, left_eigvectors, outer_products = self.calc_eigensystem(transfer_matrix, sort_order=sort_order)
        projector = self.give_projector(direction)
        approximated_inverse_prefactor = self.calc_matrix_inversion_prefactor(outer_products, projector=projector)
        
        Xs = [projector@u_v@(projector.T) for u_v in outer_products]

        Akm_n = []
        K = self.modes_per_unit_cell + 1 # M + 1
        for n in range(K):
            X_idx = self.modes_per_unit_cell - 1 + n
            Akm_n.append(- sqrt_kappa_ext @ approximated_inverse_prefactor @ Xs[X_idx] @ mu1_inv @ sqrt_kappa_ext)

        Bkm = - sqrt_kappa_ext @ approximated_inverse_prefactor @ mu1_inv @ sqrt_kappa_ext

        Ck = jnp.sum(jnp.abs(Bkm[output_idx,:])**2*(2*bath_occupations+1))
        
        summed_noise = Ck
        for n in range(K):
            for nprime in range(K):
                Dk_n_np = jnp.sum((Akm_n[n] * jnp.conjugate(Akm_n[nprime]))[output_idx,:] * (2*bath_occupations+1))
                eigval_n = eigvals[self.modes_per_unit_cell - 1 + n]
                eigval_np = eigvals[self.modes_per_unit_cell - 1 + nprime]
                r = (eigval_n * jnp.conjugate(eigval_np))
                if direction == RIGHT_TO_LEFT:
                    r = r**(-1)
                summed_noise += Dk_n_np / ( r - 1)
        
        
        return 1/(2*jnp.abs(Bkm[output_idx,input_idx])**2) * summed_noise - 1/2*(2*bath_occupations[input_idx] + 1)
        

    def check_all_constraints(self, coupling_matrix, kappa_int_matrix, kappa_ext_matrix, max_violation):
        fulfilled_constraints = []
        for c in self.all_possible_constraints:
            if np.abs(c(None, None, coupling_matrix, kappa_int_matrix, kappa_ext_matrix))**2/2 < max_violation:
                fulfilled_constraints.append(c)
        
        return arch.conditions_to_graph_characteriser(fulfilled_constraints, self.modes_per_unit_cell)

    # def prepare_operators_chain(self, chain_length):
    #     num_modes = chain_length*self.modes_per_unit_cell

    #     mode_types_chain_odd = self.mode_types[:-1]
    #     if self.mode_types[-1] == self.mode_types[0]:
    #         mode_types_chain_even = self.mode_types[:-1]
    #     else:
    #         mode_types_chain_even = [not x for x in self.mode_types[:-1]]

    #     mode_types_chain = []
    #     for chain_idx in range(chain_length):
    #         if chain_idx % 2 == 0:
    #             mode_types_chain.extend(mode_types_chain_odd)
    #         else:
    #             mode_types_chain.extend(mode_types_chain_even)

    #     operators_chain = []
    #     for idx in range(num_modes):
    #         if mode_types_chain[idx]:
    #             operators_chain.append(sym.Mode().a)
    #         else:
    #             operators_chain.append(sym.Mode().adag)

    #     return operators_chain, mode_types_chain
    
    def __prepare_mode_types__(self, mode_types):
        if mode_types == 'no_squeezing':
            self.mode_types = [True for _ in range(self.modes_per_unit_cell + 1)]
        else:
            if len(mode_types) != (self.modes_per_unit_cell + 1):
                raise ValueError('mode_types has wrong length, it has to equal modes_per_unit_cell+1')
            self.mode_types = mode_types

        # mode_types_unit_cell contains the mode types of a "standard" unit cell
        # standard means, that the 0th mode is True (an annihilation operator)
        if self.mode_types[0] is True:
            self.mode_types_unit_cell = self.mode_types[:-1]
        else:
            self.mode_types_unit_cell = [not x for x in self.mode_types[:-1]]
        
        self.operators_unit_cell = []
        for mode_type in self.mode_types_unit_cell:
            if mode_type:
                self.operators_unit_cell.append(sym.Mode().a)
            else:
                self.operators_unit_cell.append(sym.Mode().adag)

        self.operators_chain_two_unit_cells, self.mode_types_two_unit_cells = prepare_operators_chain(self.mode_types, 2)

    def __init_gabs__(self, idx1, idx2, beamsplitter=True, append=False):
        if beamsplitter:
            varname = '|g_{'
        else:
            varname = '|\\nu_{'
        
        if idx1 < 0 or idx1 >= self.modes_per_unit_cell:
            raise ValueError('index out of range')
        else:
            varname += str(idx1) + ','
        
        if idx2 >= 0 and idx2 < self.modes_per_unit_cell:
            varname += str(idx2)
        elif idx2 < 2*self.modes_per_unit_cell:
            varname += str(idx2-self.modes_per_unit_cell) + 'p'

        varname += '}|'
        new_variable = sp.Symbol(varname, real=True)

        if append and not new_variable in self.gabs:
            self.gabs.append(new_variable)

        return new_variable
    
    def __init_gphase__(self, idx1, idx2, beamsplitter=True, append=False):
        if beamsplitter:
            varname = '\mathrm{arg}(g_{'
        else:
            varname = '\mathrm{arg}(\\nu_{'
        
        if idx1 < 0 or idx1 >= self.modes_per_unit_cell:
            raise ValueError('index out of range')
        else:
            varname += str(idx1) + ','
        
        if idx2 >= 0 and idx2 < self.modes_per_unit_cell:
            varname += str(idx2)
        elif idx2 < 2*self.modes_per_unit_cell:
            varname += str(idx2-self.modes_per_unit_cell) + 'p'

        varname += '})'
        new_variable = sp.Symbol(varname, real=True)

        if append and not new_variable in self.gphases:
            self.gphases.append(new_variable)

        return new_variable
    
    def __init_Delta__(self, idx, append=False):
        if idx >= self.modes_per_unit_cell:
            raise ValueError('index out of range')
        
        new_variable = sp.Symbol('Delta%i'%idx, real=True)
        if append and not new_variable in self.Deltas:
            self.Deltas.append(new_variable)

        return new_variable
    
    def __give_coupling_element__(self, idx1, idx2, operators_chain, with_phase=True, append=False):
        op1 = operators_chain[idx1]
        op2 = operators_chain[idx2]

        idxmin, idxmax = min(idx1, idx2), max(idx1, idx2)

        # get index within unit cell
        chain_idx = idxmin // self.modes_per_unit_cell
        idxmin = idxmin - chain_idx*self.modes_per_unit_cell
        idxmax = idxmax - chain_idx*self.modes_per_unit_cell

        # detuning
        if idx1 == idx2:
            detuning = self.__init_Delta__(idxmin, append=append)
            if isinstance(op1, sym.Annihilation_operator):
                return - detuning
            else:
                return detuning

        # beamsplitter
        elif type(op1) == type(op2):
            gabs = self.__init_gabs__(idxmin, idxmax, beamsplitter=True, append=append)
            if with_phase:
                gphase = self.__init_gphase__(idxmin, idxmax, beamsplitter=True, append=append)
            else:
                gphase = sp.S(0)
            coupling = gabs * sp.exp(sp.I*gphase)
            if idx1 < idx2:
                pass
            else:
                coupling = sp.conjugate(coupling) #beamsplitter coupling matrix is Hermitian

            if isinstance(op1, sym.Annihilation_operator):
                return coupling
            else:
                return - sp.conjugate(coupling)
            
        # squeezing
        else:
            gabs = self.__init_gabs__(idxmin, idxmax, beamsplitter=False, append=append)
            if with_phase:
                gphase = self.__init_gphase__(idxmin, idxmax, beamsplitter=False, append=append)
            else:
                gphase = sp.S(0)
            coupling = gabs * sp.exp(sp.I*gphase)
            if isinstance(op1, sym.Annihilation_operator):
                return coupling
            else:
                return -sp.conjugate(coupling)

    def __give_coupling_matrix__(self, chain_length, append=False, OBC=True):
        num_modes = self.modes_per_unit_cell * chain_length

        operators_chain, _ = prepare_operators_chain(self.mode_types, chain_length)

        coupling_matrix = sp.zeros(num_modes)

        for chain_idx in range(chain_length):
            for idx_unit_cell in range(self.modes_per_unit_cell):
                idx = chain_idx * self.modes_per_unit_cell + idx_unit_cell
                if not msc.Constraint_coupling_zero(idx_unit_cell, idx_unit_cell, self.modes_per_unit_cell) in self.enforced_constraints:
                    coupling_matrix[idx, idx] = self.__give_coupling_element__(idx, idx, operators_chain=operators_chain, append=append)
            
            for idx_unit_cell1 in range(2*self.modes_per_unit_cell):
                for idx_unit_cell2 in range(2*self.modes_per_unit_cell):
                    idx1 = chain_idx * self.modes_per_unit_cell + idx_unit_cell1
                    idx2 = chain_idx * self.modes_per_unit_cell + idx_unit_cell2

                    # conditions: idx1 and idx2 do not exceed the chain (otherwise the code would start to put the coupling elements to the next, not existing, chain element)
                    # idx1 != idx2: detuning was already filled in by the code above
                    # (idx_unit_cell1 < self.modes_per_unit_cell or idx_unit_cell2 < self.modes_per_unit_cell): only fill the entries of the current unit cell, not those of already the next one
                    if idx1 < num_modes and idx2 < num_modes and idx1 != idx2 and (idx_unit_cell1 < self.modes_per_unit_cell or idx_unit_cell2 < self.modes_per_unit_cell):
                        if not msc.Constraint_coupling_zero(idx_unit_cell1, idx_unit_cell2, self.modes_per_unit_cell) in self.enforced_constraints:
                            if not msc.Constraint_coupling_phase_zero(idx_unit_cell1, idx_unit_cell2, self.modes_per_unit_cell) in self.enforced_constraints:
                                with_phase = True
                            else:
                                with_phase = False
                            coupling_matrix[idx1, idx2] = self.__give_coupling_element__(idx1, idx2, with_phase=with_phase, operators_chain=operators_chain, append=append)

        if not OBC:
            for idx_unit_cell_N in range(self.modes_per_unit_cell):
                for idx_unit_cell_1 in range(self.modes_per_unit_cell):
                    idx_N = -self.modes_per_unit_cell + idx_unit_cell_N
                    idx_1 = idx_unit_cell_1

                    if not msc.Constraint_coupling_zero(idx_unit_cell_N, idx_unit_cell_1+self.modes_per_unit_cell, self.modes_per_unit_cell) in self.enforced_constraints:
                        if not msc.Constraint_coupling_phase_zero(idx_unit_cell_N, idx_unit_cell_1+self.modes_per_unit_cell, self.modes_per_unit_cell) in self.enforced_constraints:
                            with_phase = True
                        else:
                            with_phase = False
                        coupling_matrix[idx_N, idx_1] = self.__give_coupling_element__(idx_N, idx_1, with_phase=with_phase, operators_chain=operators_chain, append=append)
                        coupling_matrix[idx_1, idx_N] = self.__give_coupling_element__(idx_1, idx_N, with_phase=with_phase, operators_chain=operators_chain, append=append)

                    

        return coupling_matrix
    
    def __jaxify_function__(self, sympy_expression):
        return sp.utilities.lambdify(self.all_variables_list, sympy_expression, modules='jax') 

    def __give_kappa_matrices__(self, chain_length):
        # internal losses

        kappa_int_matrix_diag = []
        for _ in range(chain_length):
            kappa_int_matrix_diag.extend(self.kappa_int_matrix_diag_unit_cell)
        kappa_int_matrix = sp.diag(*kappa_int_matrix_diag)

        kappas_ext_diag_chain = []
        for _ in range(chain_length):
            kappas_ext_diag_chain.extend(self.kappas_ext_diag_unit_cell)
        kappa_ext_matrix = sp.diag(*kappas_ext_diag_chain)

        return kappa_int_matrix, kappa_ext_matrix
        
    def __initialize_parameters__(self, S_target_free_symbols_init_range):

        # free parameters in S_target
        if S_target_free_symbols_init_range is None:
            self.parameters_S_target = list(self.S_target.free_symbols)
        else:
            self.parameters_S_target = list(S_target_free_symbols_init_range.keys())
        for var in self.parameters_S_target:
            if not var.is_real:
                raise Exception('variable '+var.name+' is complex, only real variables are allowed')
        
        # internal losses
        self.variables_intrinsic_losses = []
        self.kappa_int_matrix_diag_unit_cell = []
        for mode_idx in range(self.modes_per_unit_cell):
            if self.port_intrinsic_losses[mode_idx]:
                kappa_int = sp.Symbol('kappa_int%i'%mode_idx, real=True)
                self.variables_intrinsic_losses.append(kappa_int)
                self.kappa_int_matrix_diag_unit_cell.append(kappa_int)
            else:
                self.kappa_int_matrix_diag_unit_cell.append(sp.S(0))

        # extrinsic losses
        self.variables_kappas_ext = []
        self.kappas_ext_diag_unit_cell = []
        for mode_idx in range(self.modes_per_unit_cell):
            if self.kappas_free_parameters[mode_idx]:
                kappa_ext = sp.Symbol('kappa_ext%i'%mode_idx, real=True)
                self.variables_kappas_ext.append(kappa_ext)
            else:
                kappa_ext = sp.S(1)
            self.kappas_ext_diag_unit_cell.append(kappa_ext)

        self.all_variables_list = \
            self.Deltas + \
            self.gabs + \
            self.gphases + \
            self.parameters_S_target + \
            self.variables_intrinsic_losses + \
            self.variables_kappas_ext 
        self.all_variables_types = \
            [VAR_ABS]*len(self.Deltas+self.gabs) +\
            [VAR_PHASE]*len(self.gphases) + \
            [VAR_USER_DEFINED]*len(self.parameters_S_target) + \
            [VAR_INTRINSIC_LOSS]*len(self.variables_intrinsic_losses) + \
            [VAR_EXTRINSIC_LOSS]*len(self.variables_kappas_ext)
        self.S_target_free_symbols_init_range = S_target_free_symbols_init_range

        self.S_target_jax = self.__jaxify_function__(self.S_target) 
    
    def create_initial_guess(self, conditions=[], init_abs_range=None, phase_range=None, init_intrinsinc_loss_range=None, init_extrinsic_loss_range=None, num_particles=None, with_zeros=False):
        if init_abs_range is None:
            init_abs_range = [-1., 1.]
        if phase_range is None:
            phase_range = [-np.pi, np.pi]
        if init_intrinsinc_loss_range is None:
            init_intrinsinc_loss_range = INIT_INTRINSIC_LOSS_RANGE_DEFAULT
        if init_extrinsic_loss_range is None:
            init_extrinsic_loss_range = INIT_EXTRINSIC_LOSS_RANGE_DEFAULT 

        idxs_free = self.give_free_variable_idxs(conditions, rescale_variable=None)
        
        random_guess = []
        for var_idx, var_type in enumerate(self.all_variables_types):
            if var_type == VAR_ABS:
                random_guess.append(np.random.uniform(init_abs_range[0], init_abs_range[-1], size=num_particles))
            elif var_type == VAR_PHASE:
                random_guess.append(np.random.uniform(phase_range[0], phase_range[-1], size=num_particles))
            elif var_type == VAR_INTRINSIC_LOSS:
                random_guess.append(np.random.uniform(init_intrinsinc_loss_range[0], init_intrinsinc_loss_range[-1], size=num_particles))
            elif var_type == VAR_EXTRINSIC_LOSS:
                random_guess.append(np.random.uniform(init_extrinsic_loss_range[0], init_extrinsic_loss_range[-1], size=num_particles))
            elif var_type == VAR_USER_DEFINED:
                if self.S_target_free_symbols_init_range is None:
                    user_defined_range = [-np.pi, np.pi]
                else:
                    user_defined_range = self.S_target_free_symbols_init_range[self.all_variables_list[var_idx]]
                random_guess.append(np.random.uniform(user_defined_range[0], user_defined_range[-1], size=num_particles))
            else:
                raise Exception('unknown variable type')
        
        init_guess = np.array(random_guess)[idxs_free]
        if num_particles is not None:        
            init_guess = init_guess.T #sort such that particle index comes first and then the parameter index

        if not with_zeros:
            return init_guess, idxs_free
        else:
            if num_particles is not None:
                init_guess_full_array = np.zeros([num_particles, len(self.all_variables_list)])
            else:
                init_guess_full_array = np.zeros([len(self.all_variables_list)])
            init_guess_full_array[..., idxs_free] = init_guess
            init_guess_full_array[..., -1] = 1.
            return init_guess_full_array, idxs_free

    def setup_bounds(self, free_idxs, bounds_intrinsic_loss=None, bounds_extrinsic_loss=None):
        if bounds_intrinsic_loss is None and bounds_extrinsic_loss is None:
            return None

        if bounds_intrinsic_loss is None:
            bounds_intrinsic_loss = [-np.inf, np.inf]
        if bounds_extrinsic_loss is None:
            bounds_extrinsic_loss = [-np.inf, np.inf]
        
        bounds = []
        for var_type in self.all_variables_types:
            if var_type==VAR_ABS or var_type == VAR_PHASE or var_type == VAR_USER_DEFINED:
                bounds.append([-np.inf, np.inf])
            elif var_type == VAR_INTRINSIC_LOSS:
                bounds.append(bounds_intrinsic_loss)
            elif var_type == VAR_EXTRINSIC_LOSS:
                bounds.append(bounds_extrinsic_loss)
            else:
                raise NotImplementedError()
            
        return np.asarray(bounds)[free_idxs]
    
    def __initialize_conditions_func__(self, c):
        if type(c) == msc.Scaling_Rate_Constraint or type(c) == msc.Prefactor_Constraint:
            c.__initialize__(self.calc_gain_scaling_prefactor, self.calc_gain_scaling_rate)
        elif type(c) == msc.Prefactor_Bulk_Constraint:
            c.__initialize__(self.calc_gain_scaling_prefactor_bulk)
        elif isinstance(c, msc.Stability_Constraint):
            c.__initialize__(self.__provide_function_for_dynamical_matrix__(c.chain_length))
        elif isinstance(c, msc.Output_Noise_Constraint):
            c.__initialize__(self.calc_output_noise_on_resonance)
        elif isinstance(c, msc.Input_Noise_Constraint):
            c.__initialize__(self.calc_input_noise_on_resonance)
        elif isinstance(c, msc.Input_Reflection_Constraint):
            c.__initialize__(self.calc_reflection_at_input)
        elif isinstance(c, msc.Min_Distance_Eigvals):
            c.__initialize__(self.calc_eigvals_transfer_matrix, self.modes_per_unit_cell)
        else:
            return NotImplementedError()
            
    def __initialize_all_conditions_func__(self):

        self.enforced_constraints_beyond_coupling_constraint = []
        for c in self.enforced_constraints:
            if not isinstance(c, msc.Coupling_Constraint):
                self.__initialize_conditions_func__(c)
                self.enforced_constraints_beyond_coupling_constraint.append(c)

        def calc_conditions(input_array):
            if len(self.enforced_constraints_beyond_coupling_constraint) > 0:
                full_coupling_matrix = self.coupling_matrix_infinite_jax(*input_array)
                kappa_int_matrix = self.kappa_int_matrix_infinite_jax(*input_array)
                kappa_ext_matrix = self.kappa_ext_matrix_infinite_jax(*input_array)
                constraints = jnp.hstack([c(input_array, None, full_coupling_matrix, kappa_int_matrix, kappa_ext_matrix) for c in self.enforced_constraints_beyond_coupling_constraint])
                return jnp.sum(jnp.abs(constraints)**2)/2, {}
            else:
                return 0., {}

        # def coupling_matrix_effective(input_array):
        #     return self.coupling_matrix_dimensionless_jax(*input_array)
        
        # self.coupling_matrix_effective = coupling_matrix_effective

        # def calc_target_scattering_matrix(input_array):
        #     return self.S_target_jax(*input_array)
        
        # def calc_scattering_matrix_for_parameters(omega, input_array):
        #     coupling_matrix = self.coupling_matrix_effective(input_array)
        #     kappa_int_matrix = self.kappa_int_matrix_dimensionless_jax(*input_array)
        #     kappa_ext_matrix = self.kappa_ext_matrix_jax(*input_array)
        #     scattering_matrix = calc_scattering_matrix_from_coupling_matrix(coupling_matrix, kappa_int_matrix, kappa_ext_matrix, epsilon=input_array[-1], omega=omega)
        #     return scattering_matrix

        # self.calc_scattering_matrix_for_parameters = calc_scattering_matrix_for_parameters

        # def calc_conditions(input_array):
        #     scattering_matrix_target = calc_target_scattering_matrix(input_array).flatten()
        #     scattering_matrix = self.calc_scattering_matrix_for_parameters(omega=0., input_array=input_array)

        #     difference = (scattering_matrix[self.S_target_mask] - scattering_matrix_target).flatten()

        #     if len(self.enforced_constraints_beyond_coupling_constraint) > 0:
        #         full_coupling_matrix = self.coupling_matrix_dimensionless_jax(*input_array)
        #         kappa_int_matrix = self.kappa_int_matrix_dimensionless_jax(*input_array)
        #         kappa_ext_matrix = self.kappa_ext_matrix_jax(*input_array)
        #         additional_constraints = jnp.hstack([c(input_array, scattering_matrix, full_coupling_matrix, kappa_int_matrix, kappa_ext_matrix) for c in self.enforced_constraints_beyond_coupling_constraint])
        #         evaluated_conditions = jnp.hstack((jnp.real(difference), jnp.imag(difference), additional_constraints))
        #     else:
        #         evaluated_conditions = jnp.hstack((jnp.real(difference), jnp.imag(difference)))
        #     return jnp.sum(jnp.abs(evaluated_conditions)**2)/2, {'scattering_matrix': scattering_matrix, 'scattering_matrix_target': scattering_matrix_target}

        return calc_conditions
    
    def give_free_variable_idxs(self, conditions, rescale_variable=None):
        free_variable_idxs = [idx for idx in range(len(self.all_variables_list))]
        for c in conditions:
            if type(c) == msc.Constraint_coupling_zero:
                if c.idxs[0] != c.idxs[1]:
                    beamsplitter = self.mode_types_two_unit_cells[c.idxs[0]] == self.mode_types_two_unit_cells[c.idxs[1]]
                    gabs = self.__init_gabs__(c.idxs[0], c.idxs[1], beamsplitter)
                    gphase = self.__init_gphase__(c.idxs[0], c.idxs[1], beamsplitter)
                    if gabs in self.all_variables_list:
                        free_variable_idxs.remove(self.all_variables_list.index(gabs))
                    if gphase in self.all_variables_list:
                        free_variable_idxs.remove(self.all_variables_list.index(gphase))
                else:
                    Delta = self.__init_Delta__(c.idxs[0])
                    if Delta in self.all_variables_list:
                        free_variable_idxs.remove(self.all_variables_list.index(Delta))
            elif type(c) == msc.Constraint_coupling_phase_zero:
                beamsplitter = self.mode_types_two_unit_cells[c.idxs[0]] == self.mode_types_two_unit_cells[c.idxs[1]]
                gphase = self.__init_gphase__(c.idxs[0], c.idxs[1], beamsplitter)
                if gphase in self.all_variables_list:
                    free_variable_idxs.remove(self.all_variables_list.index(gphase))
            else:
                raise Exception('only architectural constraints are allowed')
            
        return free_variable_idxs

    def give_conditioned_swarm_loss_function(self, num_particles, conditions, rescale_variable=None):
        idxs_free_variables = self.give_free_variable_idxs(conditions, rescale_variable=rescale_variable)
        np_idxs_free_variables = np.array(idxs_free_variables)
        if rescale_variable is not None:
            raise NotImplementedError()

        num_total_variables = len(self.all_variables_list)
        def calc_swarm_loss_constrained(partial_input_array):
            full_input_array = np.zeros([num_particles, num_total_variables])
            full_input_array[:,idxs_free_variables] = partial_input_array
            output, _ = self.conditions_func_swarm(full_input_array)
            return output

        return calc_swarm_loss_constrained

    def give_conditions_func_with_conditions(self, conditions, rescale_variable=None):
        idxs_free_variables = self.give_free_variable_idxs(conditions, rescale_variable=rescale_variable)
        np_idxs_free_variables = np.array(idxs_free_variables)
        if rescale_variable is not None:
            idx_rescaled = self.all_variables_list.index(rescale_variable)

        num_total_variables = len(self.all_variables_list)
        def calc_conditions_constrained(partial_input_array):
            full_input_array = np.zeros(num_total_variables)
            full_input_array[idxs_free_variables] = partial_input_array
            return self.conditions_func(full_input_array)
        
        if self.gradient_method == DIFFERENCE_QUOTIENT:
            calc_jacobian = '2-point'
            calc_hessian = '2-point'
        else:
            def calc_jacobian_constrained(partial_input_array):
                full_input_array = np.zeros(num_total_variables)
                full_input_array[idxs_free_variables] = partial_input_array
                # return self.jacobian(full_input_array)[jnp.array(idxs_free_variables)]
                jacobian, aux_dict = self.jacobian(full_input_array)
                return np.array(jacobian)[np_idxs_free_variables]
            
            def calc_hessian_constrained(partial_input_array):
                raise NotImplementedError()
                # full_input_array = np.zeros(num_total_variables)
                # full_input_array[idxs_free_variables] = partial_input_array
                # if rescale_variable is None:
                #     full_input_array[-1] = 1. #epsilon=1.
                # else:
                #     full_input_array[idx_rescaled] = 1.
                # return np.take(np.take(np.array(self.hessian(full_input_array)), np_idxs_free_variables, axis=0), np_idxs_free_variables, axis=1)
            
            calc_jacobian = calc_jacobian_constrained
            calc_hessian = calc_hessian_constrained
  
        return calc_conditions_constrained, calc_jacobian, calc_hessian

    def complete_variable_arrays_with_zeros(self, variable_array, conditions, rescale_variable=None):
        free_variable_idxs = self.give_free_variable_idxs(conditions, rescale_variable=rescale_variable)
        if len(variable_array.shape) == 1:
            complete_variable_array = np.zeros(len(self.all_variables_list))
        else:
            complete_variable_array = np.zeros(list(variable_array.shape[:-1]) +[len(self.all_variables_list)])
        complete_variable_array[...,free_variable_idxs] = variable_array

        return complete_variable_array
    
    def perform_rescaling(self, unscaled_input, rescale_variable, conditions):
        raise NotImplementedError()

    def optimize_PSO_given_conditions(self,
                conditions=None, graph_characteriser=None, verbosity=False,
                kwargs_initialisation = {},
                kwargs_bounds = {},
                threshold_accept_solution=DEFAULT_THRESHOLD, 
                loss_func_swarm=None,
                pso_parameters=DEFAULT_PSO_PARAMETERS,
                log_full_swarm_history=False
            ):
        if conditions is None:
            conditions = arch.graph_characteriser_to_conditions(graph_characteriser)
        
        num_particles = pso_parameters['num_particles']

        if loss_func_swarm is None:
            loss_func_swarm = self.give_conditioned_swarm_loss_function(conditions=conditions, num_particles=num_particles)

        init_particle_positions, free_idxs = self.create_initial_guess(conditions=conditions, num_particles=num_particles, **kwargs_initialisation)

        bounds = self.setup_bounds(free_idxs, **kwargs_bounds)

        pso_optimizer = Custom_GlobalBestPSO(
            n_particles=num_particles,
            dimensions=len(free_idxs),
            options=pso_parameters['options'],
            bounds=bounds.T,
            init_pos=init_particle_positions,
            bh_strategy=pso_parameters['bh_strategy']
        )

        final_cost, best_pos = pso_optimizer.optimize(loss_func_swarm, pso_parameters['max_iterations'], interrupt_threshold=threshold_accept_solution, verbose=verbosity)

        # pso_optimizer = LocalBestPSO(
        #     n_particles=num_particles,
        #     dimensions=len(free_idxs),
        #     options=pso_parameters['options'],
        #     bounds=bounds.T,
        #     init_pos=init_particle_positions,
        #     bh_strategy=pso_parameters['bh_strategy']
        # )

        # final_cost, best_pos = pso_optimizer.optimize(loss_func_swarm, pso_parameters['max_iterations'], verbose=verbosity)

        success = final_cost < threshold_accept_solution
        solution_complete_array = self.complete_variable_arrays_with_zeros(best_pos, conditions)


        # TODO: complete rest from here

        info_out = {
            'bounds': bounds,
            'initial_guess': self.complete_variable_arrays_with_zeros(init_particle_positions, conditions),
            'free_idxs': free_idxs,
            'solution': solution_complete_array,
            'best_pos': best_pos,
            'solution_dict': create_solutions_dict(self.all_variables_list, solution_complete_array),
            'final_cost': final_cost,
            'success': success,
            'coupling_matrix': self.coupling_matrix_infinite_jax(*solution_complete_array),
            'kappa_int_matrix': self.kappa_int_matrix_infinite_jax(*solution_complete_array),
            'kappa_ext_matrix': self.kappa_ext_matrix_infinite_jax(*solution_complete_array),
            'nit': len(pso_optimizer.cost_history),
            'mean_pbest_history': pso_optimizer.mean_pbest_history,
            'loss_history': pso_optimizer.cost_history,
        }

        if log_full_swarm_history:
            # position_history = np.array(pso_optimizer.pos_history) # shape = (swarm update steps, num_particles, num_free_parameters)
            # pos_shape = position_history.shape
            # num_total_positions = pos_shape[0] * pos_shape[1]
            # num_free_parameters = pos_shape[2]
            # calc_loss_for_history = self.give_conditioned_swarm_loss_function(num_total_positions, conditions)
            # all_losses = calc_loss_for_history(position_history.reshape([num_total_positions, num_free_parameters])).reshape([pos_shape[0], pos_shape[1]])
            # pso_optimizer.particle_cost_history = all_losses
            info_out['pso_optimizer'] = pso_optimizer

        return success, info_out
    
    def repeated_optimization(self, conditions=None, graph_characteriser=None):
        return self.repeated_optimization_custom_parameters(
            conditions=conditions, 
            graph_characteriser=graph_characteriser,
            pso_parameters=self.pso_parameters,
            **self.kwargs_optimization
        )
    
    def repeated_optimization_custom_parameters(self,
                conditions=None, graph_characteriser=None, 
                num_tests=10,
                verbosity=False,
                kwargs_initialisation = {},
                kwargs_bounds = {},
                threshold_accept_solution=DEFAULT_THRESHOLD,
                interrupt_if_successful=True,
                pso_parameters=DEFAULT_PSO_PARAMETERS,
                log_full_swarm_history=False,
            ):
        
        if conditions is None:
            conditions = arch.graph_characteriser_to_conditions(graph_characteriser)

        loss_func_swarm = self.give_conditioned_swarm_loss_function(conditions=conditions, num_particles=pso_parameters['num_particles'])

        successes = []
        infos = []
        for _ in range(num_tests):
            success, info = self.optimize_PSO_given_conditions(
                conditions=conditions, graph_characteriser=graph_characteriser, verbosity=verbosity,
                kwargs_initialisation=kwargs_initialisation,
                kwargs_bounds=kwargs_bounds,
                threshold_accept_solution=threshold_accept_solution, 
                loss_func_swarm=loss_func_swarm,
                pso_parameters=pso_parameters,
                log_full_swarm_history=log_full_swarm_history
            )
            successes.append(success)
            infos.append(info)

            if success and interrupt_if_successful:
                break
        
        return np.any(successes), infos, np.where(successes)[0]

    def prepare_all_possible_combinations(self):
        indices_graph_characteriser = arch.give_graph_characterisation_indices(self.modes_per_unit_cell)

        self.possible_matrix_entries = []
        for idx1, idx2 in np.array(indices_graph_characteriser).T:
            if idx1 == idx2:
                if msc.Constraint_coupling_zero(idx1,idx1,self.modes_per_unit_cell) in self.enforced_constraints:
                    allowed_entries = [arch.NO_COUPLING]
                else:
                    allowed_entries = [arch.NO_COUPLING, arch.DETUNING]
            else:
                allowed_entries = [arch.NO_COUPLING, arch.COUPLING_WITHOUT_PHASE, arch.COUPLING_WITH_PHASE]
                if msc.Constraint_coupling_zero(idx1,idx2,self.modes_per_unit_cell) in self.enforced_constraints:
                    allowed_entries.remove(arch.COUPLING_WITHOUT_PHASE)
                    allowed_entries.remove(arch.COUPLING_WITH_PHASE)
                elif msc.Constraint_coupling_phase_zero(idx1,idx2,self.modes_per_unit_cell) in self.enforced_constraints:
                    allowed_entries.remove(arch.COUPLING_WITHOUT_PHASE)
                
                if type(self.operators_chain_two_unit_cells[idx1]) != type(self.operators_chain_two_unit_cells[idx2]):
                    if not self.phase_constraints_for_squeezing:
                        if arch.COUPLING_WITHOUT_PHASE in allowed_entries:
                            allowed_entries.remove(arch.COUPLING_WITHOUT_PHASE)

            self.possible_matrix_entries.append(allowed_entries)

        self.list_possible_graphs = []
        self.complexity_levels = []
        for p_coupl in tqdm(product(*self.possible_matrix_entries)):
            self.list_possible_graphs.append(np.array(p_coupl, dtype='int8'))
            self.complexity_levels.append(sum(p_coupl))

        self.list_possible_graphs = np.array(self.list_possible_graphs)
        self.complexity_levels = np.array(self.complexity_levels)

        # sort out all graphs where unit cells decouple
        mask_not_decoupled = np.sum(self.list_possible_graphs[:,arch.graph_characteriser_indicies_coupling_between_unit_cells(2)], -1) != 0
        print("sorted out %i graphs, where the unit cells were decoupled"%(np.sum(~mask_not_decoupled)))
        self.list_possible_graphs = self.list_possible_graphs[mask_not_decoupled]
        self.complexity_levels = self.complexity_levels[mask_not_decoupled]
        self.unique_complexity_levels = np.flip(sorted(np.unique(self.complexity_levels)))

    def find_valid_combinations(self, complexity_level, combinations_to_test=None, perform_graph_reduction_of_successfull_graphs=True):
        
        newly_added_combos = []
        
        if combinations_to_test is None:
            potential_combinations = self.identify_potential_combinations(complexity_level)
        else:
            potential_combinations = combinations_to_test

        count_tested = 0

        for combo_idx in trange(len(potential_combinations)):
            combo = potential_combinations[combo_idx]
            conditions = arch.graph_characteriser_to_conditions(combo)
            if not arch.check_if_subgraph_upper_triangle(combo, np.asarray(newly_added_combos)):
                success, all_infos, _ = self.repeated_optimization(conditions=conditions)
                count_tested += 1
                if success:
                    if perform_graph_reduction_of_successfull_graphs:
                        valid_combo_to_add = self.check_all_constraints(all_infos[-1]['coupling_matrix'], all_infos[-1]['kappa_int_matrix'], all_infos[-1]['kappa_ext_matrix'], self.kwargs_optimization['threshold_accept_solution'])
                        print('tested valid graph:', combo, 'added valid graph:', valid_combo_to_add, file=self.logfile)
                    else:
                        valid_combo_to_add = combo
                    self.valid_combinations.append(valid_combo_to_add)
                    newly_added_combos.append(valid_combo_to_add)
                else:
                    self.invalid_combinations.append(combo)
                    print('tested invalid graph:', combo, file=self.logfile)
        
        self.tested_complexities.append([complexity_level, count_tested])
    
    def identify_potential_combinations(self, complexity_level, skip_check_for_valid_subgraphs=False):
        all_idxs_with_desired_complexity = np.where(self.complexity_levels == complexity_level)[0]
        if len(all_idxs_with_desired_complexity) == 0:
            raise Warning('no architecture with the requrested complexity_level exists')
        
        potential_combinations = []
        for combo_idx in all_idxs_with_desired_complexity:
            coupling_matrix_combo = self.list_possible_graphs[combo_idx]

            #check if suggested graph is subgraph of an invalid graph
            cond1 = not arch.check_if_subgraph_upper_triangle(np.asarray(self.invalid_combinations), coupling_matrix_combo)

            if cond1:
                #check if a valid architecture is a subgraph to the suggested graph 
                if skip_check_for_valid_subgraphs:
                    cond2 = True
                else:
                    cond2 = not arch.check_if_subgraph_upper_triangle(coupling_matrix_combo, np.asarray(self.valid_combinations))
                
                if cond2:
                    potential_combinations.append(coupling_matrix_combo)

        return potential_combinations
    
    def cleanup_valid_combinations(self):
        raise NotImplementedError("use instead the function find_irreducible_graphs")
    
        all_unique_valid_combinations_array = np.unique(np.asarray(self.valid_combinations), axis=0)
        cleaned_valid_combinations = []
        num_valid_combinations = all_unique_valid_combinations_array.shape[0]

        for combo_idx, valid_combo in tqdm(enumerate(all_unique_valid_combinations_array)):
            idxs_combis_to_compare_against = np.setdiff1d(np.arange(num_valid_combinations),combo_idx)
            #check if any other of the valid architecture is a subgraph of the current architecture 
            if not arch.check_if_subgraph_upper_triangle(valid_combo, all_unique_valid_combinations_array[idxs_combis_to_compare_against]):
                cleaned_valid_combinations.append(valid_combo)
        
        self.valid_combinations = cleaned_valid_combinations

    def give_symbolic_dynamical_matrix(self, subs_dict={}):
        coupling_matrix = self.coupling_matrix_infinite.subs(subs_dict)
        kappa_int_matrix = self.kappa_int_matrix_infinite.subs(subs_dict)
        kappa_ext_matrix = self.kappa_ext_matrix_infinite.subs(subs_dict)

        dynamical_matrix = -sp.I*coupling_matrix - (kappa_ext_matrix+kappa_int_matrix)/sp.S(2)

        return dynamical_matrix

    def give_symbolic_mus(self, subs_dict={}):
        modes_per_unit_cell = self.modes_per_unit_cell

        dynamical_matrix = self.give_symbolic_dynamical_matrix(subs_dict)

        mu0 = dynamical_matrix[:modes_per_unit_cell,:modes_per_unit_cell]
        mu1 = dynamical_matrix[:modes_per_unit_cell,modes_per_unit_cell:2*modes_per_unit_cell]
        mu_1 = dynamical_matrix[modes_per_unit_cell:2*modes_per_unit_cell,:modes_per_unit_cell]

        return mu_1, mu0, mu1

    def give_symbolic_transfer_matrix(self, subs_dict={}, omega=sp.Symbol('omega', real=True)):
        modes_per_unit_cell = self.modes_per_unit_cell

        mu_1, mu0, mu1 = self.give_symbolic_mus(subs_dict)

        mu1_det = mu1.det()
        if mu1_det == sp.S(0):
            epsilon = sp.Symbol('epsilon', real=True)
            for idx in range(modes_per_unit_cell):
                if mu1[idx,idx] == sp.S(0):
                    mu1[idx,idx] = epsilon
        mu1_inv = mu1.inv()
        identity = sp.eye(modes_per_unit_cell)
        zero_matrix = sp.zeros(modes_per_unit_cell)
        transfer_matrix = sp.simplify(sp.Matrix(sp.BlockMatrix([[-mu1_inv@(mu0 + sp.I*omega*identity), -mu1_inv@mu_1], [identity, zero_matrix]])))

        infos = {'mu1_inv': mu1_inv}

        return transfer_matrix, infos
    
def find_irreducible_lattices(all_valid_lattices):
    all_unique_lattices = np.unique(np.asarray(valid_combinations), axis=0)
    irreducible_valid_lattices = []
    num_valid_combinations = all_unique_lattices.shape[0]

    for combo_idx, valid_combo in tqdm(enumerate(all_unique_lattices)):
        idxs_combis_to_compare_against = np.setdiff1d(np.arange(num_valid_combinations),combo_idx)
        #check if any other of the valid architecture is a subgraph of the current architecture 
        if not arch.check_if_subgraph_upper_triangle(valid_combo, all_unique_lattices[idxs_combis_to_compare_against]):
            irreducible_valid_lattices.append(valid_combo)
    
    return irreducible_valid_lattices