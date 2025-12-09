import jax
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
from collections import Counter
from tqdm import trange
import numpy as np
import math

from autolattice.definitions import LEFT_TO_RIGHT, RIGHT_TO_LEFT

JAX_AUTODIFF = 'JAX_AUTODIFF'
DIFFERENCE_QUOTIENT = 'DIFFERENCE_QUOTIENT'

def calc_difference_quotient(func, x0, order=1, h=1.e-2, acc=4, coeffs=None):
    if order == 0:
        result = func(x0)
    else:
        if coeffs is None:
            coeffs = get_coefficients(order, acc)
        num_coeffs = len(coeffs)
        x_range = h*(num_coeffs-1)
        result = 0
        for idx in range(num_coeffs):
            x_in = x0 + h * idx - x_range/2
            result += coeffs[idx] * func(x_in)
        result = result/h**order
        return result
    
def get_coefficients(deriv, acc):
    return np.linalg.solve(*build_equation_system(deriv, acc))

def build_equation_system(deriv, acc):
    num_coeffs, num_coeffs_left = get_num_coeffs(deriv, acc)

    lhs = np.ones((num_coeffs, num_coeffs))
    for ii in range(num_coeffs):
        lhs[:, ii] = (- num_coeffs_left + ii) ** np.arange(0, num_coeffs)

    rhs = np.zeros(num_coeffs)
    rhs[deriv] = math.factorial(deriv)

    return lhs, rhs

def get_num_coeffs(deriv, acc):
    num_coeffs = 2 * ((deriv + 1) // 2) - 1 + acc
    num_coeffs_left = (num_coeffs - 1) // 2
    return num_coeffs, num_coeffs_left

class Base_Constraint():
    def __init__(self):
        pass

    def __initialize__(self, *args, **kwargs):
        pass

    def __call__(self, input_array, S, coupling_matrix, kappa_int_matrix, kappa_ext_matrix):
        raise NotImplementedError()
    
    def __equ__(self, other_object):
        raise NotImplementedError()
    
class Min_Distance_Eigvals(Base_Constraint):
    def __init__(self, target_omega=0., min_distance=1.e-2, direction=LEFT_TO_RIGHT, punishment=1.):
        self.target_omega = target_omega
        self.min_distance = min_distance
        self.direction = direction
        self.punishment = punishment

    def __initialize__(self, calc_eigvals, modes_per_unit_cell):
        self.calc_eigvals = calc_eigvals
        self.modes_per_unit_cell = modes_per_unit_cell

    def __call__(self, input_array, S, coupling_matrix, kappa_int_matrix, kappa_ext_matrix):
        eigvals_abs = jnp.abs(self.calc_eigvals(self.target_omega, input_array, direction=self.direction))
        if self.modes_per_unit_cell > 1:
            eigval = eigvals_abs[self.modes_per_unit_cell - 1]
            eigval_previous = eigvals_abs[self.modes_per_unit_cell - 2]
            eigval_next = eigvals_abs[self.modes_per_unit_cell]
        else:
            eigval = eigvals_abs[self.modes_per_unit_cell - 1]
            eigval_next = eigvals_abs[self.modes_per_unit_cell]
            eigval_previous = eigval

        # distance_condition = (jnp.abs(eigval - eigval_next) < self.min_distance) + (jnp.abs(eigval - eigval_previous) < self.min_distance)

        # return distance_condition * self.punishment
            
        min_distance_to_next = eigval_next - eigval
        min_distance_to_previous = eigval - eigval_previous
    
        return jax.nn.relu(self.min_distance - min_distance_to_next) + jax.nn.relu(self.min_distance - min_distance_to_previous)

class Output_Noise_Constraint(Base_Constraint):
    def __init__(self, input_idx, output_idx, target_value=0.5):
        self.target_value = target_value
        self.input_idx = input_idx
        self.output_idx = output_idx

    def __initialize__(self, calc_output_noise_on_resonance):
        self.calc_output_noise_on_resonance = calc_output_noise_on_resonance

    def __call__(self, input_array, *args):
        added_noise = self.calc_output_noise_on_resonance(input_array, input_idx=self.input_idx, output_idx=self.output_idx, bath_occupations=jnp.zeros(2))
        return jnp.abs(added_noise-self.target_value)
    
class Input_Noise_Constraint(Base_Constraint):
    def __init__(self, input_idx, target_value=0.5):
        self.target_value = target_value
        self.input_idx = input_idx

    def __initialize__(self, calc_noise_on_resonance):
        self.calc_noise_on_resonance = calc_noise_on_resonance

    def __call__(self, input_array, *args):
        added_noise = self.calc_noise_on_resonance(input_array, input_idx=self.input_idx, bath_occupations=jnp.zeros(2))
        return jnp.abs(added_noise-self.target_value)
    
class Input_Reflection_Constraint(Base_Constraint):
    def __init__(self, input_idx, target_value=0., target_omega=0.):
        self.target_value = target_value
        self.input_idx = input_idx
        self.target_omega = target_omega

    def __initialize__(self, calc_reflection_at_input):
        self.calc_reflection_at_input = calc_reflection_at_input

    def __call__(self, input_array, *args):
        reflection = self.calc_reflection_at_input(self.target_omega, input_array, input_idx=self.input_idx)
        return jnp.abs(jnp.abs(reflection)-self.target_value)

class Min_Distance_Eigvals_range(Min_Distance_Eigvals):
    def __init__(self, omegas, min_distance):
        self.omegas = omegas
        self.min_distance = min_distance
    
    def __initialize__(self, calc_eigvals, modes_per_unit_cell):
        super().__initialize__(calc_eigvals, modes_per_unit_cell)
        self.calc_eigvals_vmap = jax.vmap(self.calc_eigvals, in_axes=[0,None])

    def __call__(self, input_array, *args):
        eigvals_abs = jnp.abs(self.calc_eigvals_vmap(self.omegas, input_array))
        if self.modes_per_unit_cell > 1:
            eigval = eigvals_abs[:, self.modes_per_unit_cell - 1]
            # eigval_previous = eigvals_abs[:, self.modes_per_unit_cell - 2]
            eigval_next = eigvals_abs[:, self.modes_per_unit_cell]
        else:
            raise NotImplementedError()
        
        # min_distance_to_previous = jnp.min(jnp.abs(jnp.abs(eigval - eigval_previous)))
        min_distance_to_next = jnp.min(jnp.abs(jnp.abs(eigval - eigval_next)))

        # return jax.nn.relu(self.min_distance - min_distance_to_previous) + jax.nn.relu(self.min_distance - min_distance_to_next)
        return jax.nn.relu(self.min_distance - min_distance_to_next)

class Stability_Constraint(Base_Constraint):
    def __init__(self, chain_length):
        self.chain_length = chain_length

    def __initialize__(self, calc_dynamical_matrix_OBC):
        self.calc_dynamical_matrix_OBC = calc_dynamical_matrix_OBC
    
    def __call__(self, input_array, *args):
        dynamical_matrix = self.calc_dynamical_matrix_OBC(input_array)
        eigvals = jnp.linalg.eigvals(dynamical_matrix)
        max_real_part = jnp.max(jnp.real(eigvals)) 
        return jax.nn.relu(max_real_part)
    
class Stability_Constraint_constant(Stability_Constraint):
    def __init__(self, chain_length, punishment_for_instability=1.):
        self.chain_length = chain_length
        self.punishment_for_instability = punishment_for_instability

    def __initialize__(self, calc_dynamical_matrix_OBC):
        self.calc_dynamical_matrix_OBC = calc_dynamical_matrix_OBC
    
    def __call__(self, input_array, *args):
        dynamical_matrix = self.calc_dynamical_matrix_OBC(input_array)
        eigvals = jnp.linalg.eigvals(dynamical_matrix)
        return jnp.any(jnp.real(eigvals)>0) * self.punishment_for_instability
    
    def __eq__(self, other_object):
        if type(self) == type(other_object):
            if self.chain_length == other_object.chain_length:
                return True
            else:
                return False
        else:
            return False


class Scaling_Constraint(Base_Constraint):
    def __init__(self, target_value=0., gradient_order=0, target_omega=0., gradient_method=DIFFERENCE_QUOTIENT, kwargs_difference_quotient={}, direction=LEFT_TO_RIGHT):
        self.gradient_order = gradient_order
        self.target_value = target_value
        self.target_omega = target_omega
        self.gradient_method = gradient_method
        self.direction = direction

        self.kwargs_difference_quotient = {'h': 1.e-2, 'acc': 4}
        self.kwargs_difference_quotient.update(kwargs_difference_quotient)
        self.kwargs_difference_quotient['order'] = gradient_order

        if self.gradient_method == JAX_AUTODIFF and gradient_order != 0:
            raise NotImplementedError('jax autodiff is not implemented for eigenvectors, use difference quotient instead')
        
        self.func_to_call = None

    def __initialize__(self, calc_scaling_prefactor, calc_scaling_rate):
        self.calc_scaling_prefactor = calc_scaling_prefactor
        self.calc_scaling_rate = calc_scaling_rate
        # self.grad_function = None
        # raise NotImplementedError()
    
    def __call__(self, input_array, *args, **kwargs):
        if self.gradient_order == 0 or self.gradient_method == JAX_AUTODIFF:
            return self.func_to_call(self.target_omega, input_array) - self.target_value
        elif self.gradient_method == DIFFERENCE_QUOTIENT:
            function_to_differentiate = lambda omega: self.func_to_call(omega, input_array)
            return calc_difference_quotient(function_to_differentiate, self.target_omega, **self.kwargs_difference_quotient) - self.target_value
        else:
            raise NotImplementedError()
    
    def __equ__(self, other_object):
        if type(self) == type(other_object):
            cond1 = self.gradient_order == other_object.gradient_order
            cond2 = self.target_value == other_object.target_value
            cond3 = self.target_omega == other_object.target_omega
            cond4 = self.direction == other_object.direction

            if cond1 and cond2 and cond3 and cond4:
                return True
        else:
            return False

class Scaling_Rate_Constraint(Scaling_Constraint):
    def __init__(self, target_value=0., gradient_order=0, target_omega=0., gradient_method=DIFFERENCE_QUOTIENT, kwargs_difference_quotient={}, direction=LEFT_TO_RIGHT, idx_eigval=None):
        self.idx_eigval = idx_eigval
        super().__init__(
            target_value=target_value,
            gradient_order=gradient_order,
            target_omega=target_omega,
            gradient_method=gradient_method,
            kwargs_difference_quotient=kwargs_difference_quotient,
            direction=direction
        )

    def __initialize__(self, _, calc_scaling_rate):
        super().__initialize__(calc_scaling_prefactor=None, calc_scaling_rate=calc_scaling_rate)

        self.func_to_call = lambda omega, input_array: jnp.abs(calc_scaling_rate(omega, input_array, idx_eigval=self.idx_eigval, direction=self.direction))

        if self.gradient_order != 0:
            if self.gradient_method == JAX_AUTODIFF:
                for _ in range(self.gradient_order):
                    self.func_to_call = jax.grad(self.func_to_call)
            elif self.gradient_method == DIFFERENCE_QUOTIENT:
                self.kwargs_difference_quotient['coeffs'] = get_coefficients(self.gradient_order, self.kwargs_difference_quotient['acc'])
            else:
                raise NotImplementedError()
        
    def __call__(self, input_array, *args, **kwargs):
        return super().__call__(input_array)

    def __equ__(self, other_object):
        if super().__equ__(other_object):
            if self.idx_eigval == other_object.idx_eigval:
                return True
        
        return False
        
class Prefactor_Constraint(Scaling_Constraint):
    def __init__(self, target_value=0., gradient_order=0, target_omega=0., gradient_method=DIFFERENCE_QUOTIENT, kwargs_difference_quotient={}, direction=LEFT_TO_RIGHT, input_idx=0, output_idx=0):
        self.input_idx = input_idx
        self.output_idx = output_idx
        super().__init__(
            target_value=target_value,
            gradient_order=gradient_order,
            target_omega=target_omega,
            gradient_method=gradient_method,
            kwargs_difference_quotient=kwargs_difference_quotient,
            direction=direction
        )

    def __initialize__(self, calc_scaling_prefactor, calc_scaling_rate):
        super().__initialize__(calc_scaling_prefactor=calc_scaling_prefactor, calc_scaling_rate=None)
        # if self.gradient_order == 0:
        self.func_to_call = lambda omega, input_array: jnp.abs(calc_scaling_prefactor(omega, input_array, direction=self.direction)[self.output_idx, self.input_idx])
        if self.gradient_order != 0:
            if self.gradient_method == JAX_AUTODIFF:
                raise NotImplementedError()
            elif self.gradient_method == DIFFERENCE_QUOTIENT:
                # self.func_to_call = None
                self.kwargs_difference_quotient['coeffs'] = get_coefficients(self.gradient_order, self.kwargs_difference_quotient['acc'])
            else:
                raise NotImplementedError()
    
    def __call__(self, input_array, *args, **kwargs):
        return super().__call__(input_array)
    
    def __equ__(self, other_object):
        if super().__equ__(other_object):
            if self.input_idx == other_object.input_idx and self.output_idx == other_object.output_idx:
                return True
        
        return False
        
class Prefactor_Bulk_Constraint(Prefactor_Constraint):    
    def __initialize__(self, calc_scaling_prefactor_bulk):
        super().__initialize__(calc_scaling_prefactor=calc_scaling_prefactor_bulk, calc_scaling_rate=None)

class Coupling_Constraint(Base_Constraint):
    def __init__(self, idx1, idx2, modes_per_unit_cell=None):
        def transform_idx(idx):
            # returns the idx as index in chain notation (0,1,...,0p,1p,...) and the index of the corresponding element in the coupling matrix (1p -> 1+modes_per_unit_cell)
            if isinstance(idx, str) and idx[-1] == 'p':
                    if int(idx[:-1]) >= modes_per_unit_cell or int(idx[:-1]) < 0:
                        raise ValueError('index is out of range')
                    return idx, int(idx[:-1])+modes_per_unit_cell
            else:
                idx = int(idx) 
                if idx > 2*modes_per_unit_cell or idx < 0:
                    raise ValueError('index is out of range')
                elif idx >= modes_per_unit_cell:
                    chain_idx = str(idx - modes_per_unit_cell) + 'p'
                else:
                    chain_idx = str(idx)
                return chain_idx, idx

        self.modes_per_unit_cell = modes_per_unit_cell
        chain_idx1, idx1 = transform_idx(idx1)
        chain_idx2, idx2 = transform_idx(idx2)        

        if idx1 <= idx2:
            self.idxs = [idx1, idx2]   
            if modes_per_unit_cell is not None:
                self.chain_idxs = [chain_idx1, chain_idx2]
        else:
            self.idxs = [idx2, idx1]   
            if modes_per_unit_cell is not None:
                self.chain_idxs = [chain_idx2, chain_idx1]

        if modes_per_unit_cell is None:
            self.chain_idxs = None
        
    def __eq__(self, other_object):
        if type(self) == type(other_object):
            if set(self.idxs) == set(other_object.idxs):
                return True
            else:
                return False
        else:
            return False
    
class Constraint_coupling_zero(Coupling_Constraint):
    def __call__(self, input_array, S, coupling_matrix, kappa_int_matrix, kappa_ext_matrix):
        idx1, idx2 = self.idxs
        element = coupling_matrix[idx1, idx2]

        if idx1 == idx2:
            return jnp.array([jnp.real(element)])
        else:
            return jnp.array([jnp.abs(element)])

    def __str__(self):
        return 'Coupling between %i and %i is set to 0'%(self.idxs[0], self.idxs[1])
    
    def __hash__(self):
        return hash(('Constraint_coupling_zero', self.idxs[0], self.idxs[1]))
        
class Constraint_coupling_phase_zero(Coupling_Constraint):
    def __init__(self, idx1, idx2, modes_per_unit_cell=None):
        if idx1 == idx2:
            raise Exception('Constraint_coupling_phase_zero does only work if idx1 and idx2 are different')
        super().__init__(idx1, idx2, modes_per_unit_cell)

    def __call__(self, input_array, S, coupling_matrix, kappa_int_matrix, kappa_ext_matrix):
        idx1, idx2 = self.idxs
        element = coupling_matrix[idx1, idx2]
        return jnp.array([jnp.imag(element)])
    
    def __str__(self):
        return 'Coupling phase between %i and %i is set to 0'%(self.idxs[0], self.idxs[1])
    
    def __hash__(self):
        return hash(('Constraint_coupling_zero', self.idxs[0], self.idxs[1]))


class Equal_Coupling_Rates(Base_Constraint):
    def __init__(self, list_equal_couplings):
        self.list_equal_couplings = list_equal_couplings
    
    def __call__(self, input_array, S, coupling_matrix, kappa_int_matrix, kappa_ext_matrix):
        idx0_ref = self.list_equal_couplings[0][0]
        idx1_ref = self.list_equal_couplings[0][1]
        deviation = 0
        for idx in range(1, len(self.list_equal_couplings)):
            idx0, idx1 = self.list_equal_couplings[idx]
            deviation += jnp.abs(coupling_matrix[idx0,idx1]) - jnp.abs(coupling_matrix[idx0_ref,idx1_ref])

        return deviation
    

    