import numpy as np
import sympy as sp
import os
import matplotlib.pyplot as plt
from sympy.parsing.sympy_parser import parse_expr
from IPython.display import display, Math
from sympy.utilities.lambdify import lambdify
from sympy import N

def get_symbolic_expr_error_own(data,expr):
    # this function is a copy of get_symbolic_expr_error aifeynman.S_run_aifeynman
    # removed all try/except statements, also removed the issue of calculating failure when function is constant
    N_vars = len(data[0])-1
    possible_vars = ["x%s" %i for i in np.arange(0,30,1)]
    variables = []
    for i in range(N_vars):
        variables = variables + [possible_vars[i]]
    eq = parse_expr(expr)
    f = lambdify(variables, N(eq), modules='numpy')
    real_variables = []

    for i in range(len(data[0])-1):
        check_var = "x"+str(i)
        if check_var in np.array(variables).astype('str'):
            real_variables = real_variables + [data[:,i]]

    # Remove accidental nan's
    good_idx = ~np.isnan(f(*real_variables))

    # use this to get rid of cases where the loss gets complex because of transformations of the output variable
    # return np.mean(np.log2(1+abs(f(*real_variables)[good_idx]-data[good_idx][:,-1])*2**30))
    return float(np.mean(np.log2(1+abs(f(*real_variables)-data[:,-1])[good_idx]*2**30)))

def lambidify_expr(var_names, expr):
    all_variables = [create_variable(var) for var in var_names]
    return lambdify(all_variables, expr, modules='numpy')

def create_variable(var_name):
    if var_name == 'target_scaling_rate_squared':
        var_name = 'G'
    if var_name == 'distance_next_eigenvalue_at_0':
        var_name = 'D_0'
    if var_name == 'min_distance_next_eigenvalue':
        var_name = 'D_{min}'
    if var_name == 'bandwidth':
        var_name = 'B'
    return sp.Symbol(var_name)

def load_data(folder, var_name, idx_chosen=None):
    input_var_names, subs_input_variables = read_variable_order(folder, var_name)

    filename_solutions = os.path.join(folder, var_name, 'results/solution_dataset.txt')
    data = np.loadtxt(filename_solutions, dtype='str')
    loaded_data = {
        'average_error_in_bits': data[:,0].astype('float'),
        'complexity': data[:,-3].astype('float'),
        'mdl_loss': data[:,-2].astype('float'),
        'raw_expressions': [parse_expr(expr) for expr in data[:,-1]],
        'dataset': np.loadtxt(os.path.join(folder, var_name, 'dataset.txt')),
        'input_var_names': input_var_names,
        'target_var_name': var_name,
        'subs_input_variables': subs_input_variables
    }
    loaded_data['expressions'] = [expr.subs(subs_input_variables) for expr in loaded_data['raw_expressions']]
    loaded_data['loss'] = [get_symbolic_expr_error_own(loaded_data['dataset'], str(expr)) for expr in loaded_data['raw_expressions']]

    if idx_chosen is None:
        print('You have not chosen any expression yet, here are all of them')
        for idx in range(len(data[:,0])):
            print_chosen_expression(loaded_data, var_name, idx, input_var_names)
        return None, loaded_data
    else:
        print('You have chosen the expression:')
        print_chosen_expression(loaded_data, var_name, idx_chosen, input_var_names)
        loaded_data['chosen_expression'] = loaded_data['expressions'][idx_chosen]
        
        dependent_variable_names = []
        for var_name in input_var_names:
            variable = create_variable(var_name)
            if variable in loaded_data['chosen_expression'].free_symbols:
                dependent_variable_names.append(variable.name)

        loaded_data['dependent_variable_names'] = dependent_variable_names
        loaded_data['function'] = lambidify_expr(dependent_variable_names, loaded_data['chosen_expression'])
        return loaded_data['function'], loaded_data
        # return lambdify(full_var_list, loaded_data['expressions'][idx_chosen], modules='numpy')


def read_variable_order(folder, var_name):
    with open(os.path.join(folder, var_name, 'dataset.txt'), "r") as f:
        for line in f:
            break
        var_names = line[2:-2].split('\t')
        var_names = var_names[:-1] # cut variable to calculate from list (its the last element of the list)
    subs_dict = {}
    for count, var_name in enumerate(var_names):
        # if var_name == 'gain_rate':
        #     var_name = 'G'
        x = create_variable('x%i'%count)
        var = create_variable(var_name)
        subs_dict[x] = var
    return var_names, subs_dict

def test_expression(function, training_dataset):
    input_array = (training_dataset[:,:-1]).T
    target_output = training_dataset[:,-1]
    output = function(*input_array)
    deviation = output - target_output
    mask = ~np.isnan(output)

    nan_rate = 1 - np.mean(mask)

    # mean_deviation = np.mean(np.abs(deviation))
    # mean_relative_deviation = np.mean(np.abs(deviation[mask])/np.abs(target_output[mask]))

    MSE = np.mean(deviation[mask]**2)

    return MSE, nan_rate

def print_chosen_expression(loaded_data, var_name, idx_chosen, var_names):
    function = lambidify_expr(var_names, loaded_data['expressions'][idx_chosen])
    MSE, nan_rate = test_expression(function, loaded_data['dataset'])
    print('index: %i, loss: %0.2f, complexity: %0.2f, nan_rate: %0.2f, average_error_in_bits: %0.4f'%(idx_chosen, loaded_data['loss'][idx_chosen], loaded_data['complexity'][idx_chosen], nan_rate, loaded_data['average_error_in_bits'][idx_chosen]))
    # if var_name == 'gain_rate':
    #     var_name_to_use = 'G'
    # else:
    #     var_name_to_use = var_name
    lhs = sp.latex(create_variable(var_name))
    rhs = sp.latex(loaded_data['expressions'][idx_chosen])
    display(Math(lhs + '=' + rhs))