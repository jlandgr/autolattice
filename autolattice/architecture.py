import numpy as np
import jax
import jax.numpy as jnp
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from autolattice.constraints import Constraint_coupling_zero, Constraint_coupling_phase_zero
from autolattice.chain_functions import prepare_operators_chain

# def matrix1_subgraph_to_matrix2(matrix1, matrix2):
#     return jnp.sum((matrix2 - matrix1) < 0) == 0

# matrix1_subgraph_to_matrix2 = jax.jit(matrix1_subgraph_to_matrix2)

NO_COUPLING = 0
DETUNING = 1
COUPLING_WITHOUT_PHASE = 1
COUPLING_WITH_PHASE = 2

default_kwargs_noise = {'width_factor': 0.85, 'distance': 0.12, 'opening_angles': [65,70,75], 'base_width_radius_factor': 2.7, 'width_factor': 0.85, 'linewidth': 2}

import networkx as nx
import matplotlib.colors as colors

def length_graph_characteriser(modes_per_unit_cell):
    return int(3/2 * modes_per_unit_cell**2 + modes_per_unit_cell/2)

def conditions_to_graph_characteriser(conditions, modes_per_unit_cell):
    graph_characteriser = np.zeros(length_graph_characteriser(modes_per_unit_cell), dtype='int8')
    indices = give_graph_characterisation_indices(modes_per_unit_cell)

    idx_counter = 0
    for idx1, idx2 in np.array(indices).T:
        if Constraint_coupling_zero(idx1, idx2, modes_per_unit_cell) in conditions:
            graph_characteriser[idx_counter] = NO_COUPLING
        elif idx1 != idx2 and Constraint_coupling_phase_zero(idx1, idx2, modes_per_unit_cell) in conditions:
            graph_characteriser[idx_counter] = COUPLING_WITHOUT_PHASE
        else:
            if idx1 != idx2:
                graph_characteriser[idx_counter] = COUPLING_WITH_PHASE
            else:
                graph_characteriser[idx_counter] = DETUNING
        idx_counter += 1
    return graph_characteriser

def graph_characteriser_to_conditions(graph_characteriser):
    modes_per_unit_cell = modes_per_unit_cell_from_graph_characteriser(graph_characteriser)
    indices_adjacency_matrix = give_graph_characterisation_indices(modes_per_unit_cell)

    conditions = []
    for idx1, idx2, val_coupling in zip(indices_adjacency_matrix[0], indices_adjacency_matrix[1], graph_characteriser):
        if idx1 == idx2:
            if val_coupling == NO_COUPLING:
                conditions.append(Constraint_coupling_zero(idx1,idx2,modes_per_unit_cell))
            elif val_coupling == DETUNING:
                pass
            else:
                raise NotImplementedError()
        else:
            if val_coupling == NO_COUPLING:
                conditions.append(Constraint_coupling_zero(idx1,idx2,modes_per_unit_cell))
            elif val_coupling == COUPLING_WITHOUT_PHASE:
                conditions.append(Constraint_coupling_phase_zero(idx1,idx2,modes_per_unit_cell))
            elif val_coupling == COUPLING_WITH_PHASE:
                pass
            else:
                raise NotImplementedError()
    
    return conditions

def give_graph_characterisation_indices(modes_per_unit_cell):
    indices = []
    for idx2 in range(modes_per_unit_cell):
        for idx1 in range(idx2, 2*modes_per_unit_cell):
            indices.append([idx1, idx2])
    indices = np.asarray(indices)
    return (indices[:,1], indices[:,0])

# def give_indicies_coupling_between_unit_cells(modes_per_unit_cell, coupling_range=1):
#     if coupling_range != 1:
#         raise NotImplementedError()
#     indices = []
#     for idx2 in range(modes_per_unit_cell):
#         for idx1 in range(modes_per_unit_cell, 2*modes_per_unit_cell):
#             indices.append([idx1, idx2])
#     indices = np.asarray(indices)
#     return (indices[:,1], indices[:,0])

def graph_characteriser_indicies_coupling_between_unit_cells(modes_per_unit_cell, coupling_range=1):
    counter = 0
    idxs_list = []
    for idx1, idx2 in np.array(give_graph_characterisation_indices(modes_per_unit_cell)).T:
        if idx1 < modes_per_unit_cell and idx2 >= modes_per_unit_cell:
            idxs_list.append(counter)
        counter += 1
    return idxs_list


def modes_per_unit_cell_from_graph_characteriser(graph_characteriser):
    return (-1 + int(np.sqrt(1+24*len(graph_characteriser))))//6

def characteriser_to_adjacency_matrix(graph_characteriser):
    modes_per_unit_cell = modes_per_unit_cell_from_graph_characteriser(graph_characteriser)
    adjacency_matrix = np.zeros([2*modes_per_unit_cell, 2*modes_per_unit_cell], dtype='int8')
    mask = give_graph_characterisation_indices(modes_per_unit_cell)

    for idx1, idx2, element in zip(mask[0], mask[1], graph_characteriser):
        adjacency_matrix[idx1, idx2] = element
        adjacency_matrix[idx2, idx1] = element
        if idx1 < modes_per_unit_cell and idx2 < modes_per_unit_cell:
            adjacency_matrix[idx1+modes_per_unit_cell, idx2+modes_per_unit_cell] = element
            adjacency_matrix[idx2+modes_per_unit_cell, idx1+modes_per_unit_cell] = element
    
    return adjacency_matrix

def chain_layout(adjacency_matrix, chain_length, positions_unit_cell='straight', distance_next_unit_cell=None):
    modes_per_unit_cell = adjacency_matrix.shape[0]//2
    num_total_modes = modes_per_unit_cell * chain_length

    G_unit_cell = nx.Graph()
    for idx in range(modes_per_unit_cell):
        G_unit_cell.add_node(idx)
    for idx1 in range(modes_per_unit_cell):
        for idx2 in range(idx1+1):
            if adjacency_matrix[idx1, idx2] != 0:
                G_unit_cell.add_edge(idx1, idx2)
    
    if positions_unit_cell == 'spring_layout':
        positions_unit_cell = nx.drawing.spring_layout(G_unit_cell)
    elif positions_unit_cell == 'straight':
       positions_unit_cell = {idx: np.array([0., idx]) for idx in range(modes_per_unit_cell)}
    else:
        positions_unit_cell = {idx: np.array(pos) for idx, pos in enumerate(positions_unit_cell)}
        # raise NotImplementedError()

    all_distances_unit_cell = []
    for idx1 in range(modes_per_unit_cell):
        for idx2 in range(idx1+1, modes_per_unit_cell):
            all_distances_unit_cell.append( np.linalg.norm(positions_unit_cell[idx1] - positions_unit_cell[idx2]) )
    
    if distance_next_unit_cell is None:
        if modes_per_unit_cell > 1:
            distance_next_unit_cell = np.max(all_distances_unit_cell)
        else:
            distance_next_unit_cell = 1.

    positions = {}
    for idx in range(num_total_modes):
        idx_chain = idx // modes_per_unit_cell
        idx_unit_cell = idx % modes_per_unit_cell
        displacement = np.array([1., 0]) * distance_next_unit_cell*idx_chain
        positions.update({idx: positions_unit_cell[idx_unit_cell]+displacement})
    return positions

def reduce_color_saturation(color, color_conversion=np.array([1,0.5,1.5])):
    #color conversion factor in hsv range
    new_hsv = color_conversion*colors.rgb_to_hsv(colors.to_rgb(color))
    if new_hsv[-1] > 1:
        new_hsv[-1] = 1
    return colors.hsv_to_rgb(new_hsv)

def draw_chain_cells(list_of_graph_characteriser, size_per_column=2.5, size_per_row=2.5, architectures_per_row=5, **kwargs):

    num_columns = max(len(list_of_graph_characteriser)%architectures_per_row, architectures_per_row)
    num_rows = (len(list_of_graph_characteriser)-1)//architectures_per_row + 1
    fig, axes = plt.subplots(num_rows, num_columns, figsize=(size_per_column*num_columns,size_per_row*num_rows))
    for ax in axes.flatten():
        ax.axis('off')
    for idx, combo in enumerate(list_of_graph_characteriser):
        draw_chain_cell(combo, ax=axes.flatten()[idx], **kwargs)

def set_axes_limits(ax, pos, node_size, scale):
    """
    Adjusts xlim and ylim to account for node sizes in data coordinates.
    
    Parameters:
    - ax: The Matplotlib axis object.
    - pos: Dictionary of node positions.
    - node_size: Node size in points².
    
    This function was written by ChatGPT
    """
    x_vals = [pos[n][0] for n in pos]
    y_vals = [pos[n][1] for n in pos]
    
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)

    node_radius_data = convert_node_size_to_inch(node_size, scale)

    # Adjust limits to include node radius
    ax.set_xlim(x_min - node_radius_data, x_max + node_radius_data)
    ax.set_ylim(y_min - node_radius_data, y_max + node_radius_data)

def draw_chain_cell(
        graph_characteriser,
        mode_types='no_squeezing',
        color_detuning='black', color_passive='black', color_active='#21EF00', color_squeezing='#107DFF', #color_active='green', color_squeezing='blue',
        node_colors_unit_cell=None, positions=None, positions_unit_cell='straight',
        ax=None,
        num_previous_unit_cells=0,
        num_following_unit_cells=1,
        kwargs_draw={},
        distance_next_unit_cell=None,
        scale=0.85, # distance 1 in plot corresponds to <scale> inches
        filename=None,
        orientation_detunings=None,
        noise_orientations=None,
        kwargs_noise={},
        list_of_kwargs_noise=None,
        modes_without_noise=[]
        ):
        

    # kwargs_draw = {'edgecolors': 'black', 'linewidths': 4, 'width': 9, 'node_size': 1000, 'detuning_offset_factor': 1., 'detuning_radius_factor': 1.} | kwargs_draw

    chain_length = num_previous_unit_cells + 1 + num_following_unit_cells

    color_wheel = ['orange', 'red', 'gray', 'purple']

    adjacency_matrix = characteriser_to_adjacency_matrix(graph_characteriser)
    modes_per_unit_cell = modes_per_unit_cell_from_graph_characteriser(graph_characteriser)

    num_total_modes = modes_per_unit_cell * chain_length

    if mode_types == 'no_squeezing':
        mode_types = np.ones(modes_per_unit_cell+1, dtype='bool')
    
    if len(mode_types)-1 != modes_per_unit_cell:
        raise ValueError('length of mode types has to equal modes_per_unit_cell+1')

    _, mode_types_all = prepare_operators_chain(mode_types, chain_length)
    
    if node_colors_unit_cell is None:
        node_colors_unit_cell = [color_wheel[i] for i in range(modes_per_unit_cell)] 
        # for idx in range(len(node_colors)):
    node_colors = node_colors_unit_cell * chain_length
            # node_colors.append(reduce_color_saturation(node_colors[idx]))
    
    if orientation_detunings is not None:
        orientation_detunings = orientation_detunings * chain_length
    
    if noise_orientations is not None:
        noise_orientations = noise_orientations*chain_length
    
    for mode_idx in modes_without_noise:
        noise_orientations[mode_idx] = None

    
    if list_of_kwargs_noise is not None:
        list_of_kwargs_noise = list_of_kwargs_noise * chain_length

    G = nx.Graph()
    G.add_nodes_from([(idx, {"color": node_colors[idx]}) for idx in range(num_total_modes)])

    for idx1 in range(2*modes_per_unit_cell):
        for idx2 in range(idx1+1):
            color = None
            if idx1 == idx2:
                if adjacency_matrix[idx1, idx2] == DETUNING:
                    color = color_detuning
            elif mode_types_all[idx1] == mode_types_all[idx2]:
                if adjacency_matrix[idx1, idx2] == COUPLING_WITHOUT_PHASE:
                    color = color_passive
                elif adjacency_matrix[idx1, idx2] == COUPLING_WITH_PHASE:
                    color = color_active
            else:
                if adjacency_matrix[idx1, idx2] == COUPLING_WITH_PHASE or adjacency_matrix[idx1, idx2] == COUPLING_WITHOUT_PHASE:
                    color = color_squeezing
            
            if color is not None:
                for idx_unit_cell in range(chain_length):
                    idx1_in_unit_cell = idx_unit_cell * modes_per_unit_cell + idx1
                    idx2_in_unit_cell = idx_unit_cell * modes_per_unit_cell + idx2

                    if idx1_in_unit_cell < num_total_modes and idx2_in_unit_cell < num_total_modes:
                        G.add_edge(idx1_in_unit_cell, idx2_in_unit_cell, color=color)

    if positions is None:
        positions = chain_layout(adjacency_matrix, chain_length, positions_unit_cell=positions_unit_cell, distance_next_unit_cell=distance_next_unit_cell)

    fig, ax = draw_graph(
        G=G,
        positions=positions,
        color_detuning=color_detuning,
        node_colors=node_colors,
        scale=scale,
        filename=filename,
        orientation_detunings=orientation_detunings,
        noise_orientations=noise_orientations,
        ax=ax,
        kwargs_noise=kwargs_noise,
        list_of_kwargs_noise=list_of_kwargs_noise,
        **kwargs_draw
    )

    return fig, ax, G
    
def draw_graph(
        G,
        positions,
        node_colors,
        color_detuning='black',
        scale=0.85,
        filename=None,
        ax=None,
        orientation_detunings=None,
        edgecolors='black',
        linewidths=4,
        width=9,
        node_size=1000,
        detuning_offset_factor=1,
        detuning_radius_factor=1,
        noise_orientations=None,
        detuning_width_factor=0.8,
        kwargs_noise={},
        list_of_kwargs_noise=None,
        input_idx=0,
        output_idx=0
    ):

    kwargs_noise = default_kwargs_noise | kwargs_noise

# kwargs_draw = {'edgecolors': 'black', 'linewidths': 4, 'width': 9, 'node_size': 1000, 'detuning_offset_factor': 1., 'detuning_radius_factor': 1., 'detuning_width_factor': 0.8}

    num_nodes = G.number_of_nodes()

    if orientation_detunings is None:
        orientation_detunings = [0.] * num_nodes

    if noise_orientations is None:
        noise_orientations = [None] * num_nodes

    if list_of_kwargs_noise is None:
        list_of_kwargs_noise = [kwargs_noise] * num_nodes
    else:
        list_of_kwargs_noise = [default_kwargs_noise | el for el in list_of_kwargs_noise]

    if ax is None:
        fig, ax = plt.subplots()
        axes_was_generated_here = True
    else:
        axes_was_generated_here = False
    
    self_loops = []
    normal_edges = []
    normal_edge_colors = []
    all_edge_colors = nx.get_edge_attributes(G, 'color')#.values()
    for edge_idx, edge in enumerate(G.edges):
        if edge[0] == edge[1]:
            self_loops.append(edge)
        else:
            normal_edges.append(edge)        
            normal_edge_colors.append(all_edge_colors[edge])
    
    # Draw normal edges (edges between different nodes)
    nx.draw_networkx_edges(G, positions, edgelist=normal_edges, edge_color=normal_edge_colors, ax=ax, width=width)
    
    node_radius_data = convert_node_size_to_inch(node_size, scale)
    # node_radius_pts = np.sqrt(node_size) / 2  # Approximate radius in points
    # # Transform from points to data units
    # node_radius_data = ax.transData.inverted().transform(
    #     ax.transAxes.transform((0, node_radius_pts / 72.0))  # Convert points to inches
    # )[1] - ax.transData.inverted().transform(ax.transAxes.transform((0, 0)))[1]

    for edge in self_loops:
        node = edge[0]  # self-loop only has one node
        loop_pos = positions[node] + node_radius_data * detuning_offset_factor * np.array([np.sin(orientation_detunings)[node], np.cos(orientation_detunings)[node]])
        
        # Create a circular loop around the node
        loop = plt.Circle(
            loop_pos, 
            node_radius_data * detuning_radius_factor, 
            color=color_detuning,
            fill=False, 
            linewidth=width*detuning_width_factor,
        )
        # Add the loop as a circle to the plot
        ax.add_patch(loop)

    
    for node_idx in range(num_nodes):
        kwargs_noise = list_of_kwargs_noise[node_idx]
        num_noise_circles = len(kwargs_noise['opening_angles'])
        base_height = node_radius_data * kwargs_noise['base_width_radius_factor']
        noise_circles_heights = np.array([base_height+circle_idx*kwargs_noise['distance'] for circle_idx in range(num_noise_circles)])
        noise_circles_widths = noise_circles_heights * kwargs_noise['width_factor']
        if noise_orientations[node_idx] is not None:
            for circle_idx in range(num_noise_circles):
                opening_angle = kwargs_noise['opening_angles'][circle_idx]
                noice_circle = patches.Arc(
                    positions[node_idx],
                    width=noise_circles_widths[circle_idx], height=noise_circles_heights[circle_idx],
                    angle=-noise_orientations[node_idx]*180/np.pi,
                    theta1=90-opening_angle/2, theta2=90+opening_angle/2,
                    color='black',
                    linewidth=kwargs_noise['linewidth']
                )
                ax.add_patch(noice_circle)

    # Draw the nodes with their respective colors
    nx.draw_networkx_nodes(G, positions, node_color=node_colors, ax=ax, edgecolors=edgecolors, linewidths=linewidths, node_size=node_size)

    # kwargs_draw = {'edgecolors': 'black', 'linewidths': 3, 'width': 8, 'node_size': 1000} | kwargs_draw
        
    set_axes_limits(ax, positions, node_size*6, scale)

    ylim = ax.get_ylim()
    xlim = ax.get_xlim()
    height = ylim[1] - ylim[0]
    width = xlim[1] - xlim[0]
    
    if axes_was_generated_here:
        ax.figure.set_size_inches(width*scale, height*scale)
        fig.tight_layout(pad=0)
    else:
        fig = ax.figure

    for spine in ax.spines.values():
        spine.set_visible(False)  # Hide all spines (borders)
    
    if filename is not None:
        fig.savefig(filename, transparent=True)

    return fig, ax

def convert_node_size_to_inch(node_size, scale):
    node_radius_pts = np.sqrt(node_size) / 2  # Approximate radius in points
    node_radius_inch = node_radius_pts/72.

    # distance 1 = <scale> inches

    return node_radius_inch /scale



def characterize_architectures(list_of_architectures):
    
    list_of_detunings = []
    list_of_passive_couplings = []
    list_of_active_couplings = []
    for arch_idx in tqdm.trange(len(list_of_architectures)):
        arch = list_of_architectures[arch_idx]
        num_detunings, num_passive, num_active = characterize_architecture(arch)
        list_of_detunings.append(num_detunings)
        list_of_passive_couplings.append(num_passive)
        list_of_active_couplings.append(num_active)
    info_dict = {
        'num_detunings': np.asarray(list_of_detunings),
        'num_passive_couplings': np.asarray(list_of_passive_couplings),
        'num_active_couplings': np.asarray(list_of_active_couplings),
        'num_couplings': np.asarray(list_of_passive_couplings)+np.asarray(list_of_active_couplings)
    }
    return info_dict

def characterize_architecture(arch):
    size_upper_triangle_matrix = len(arch)
    num_modes = int((-1 + np.sqrt(1+4*2*size_upper_triangle_matrix))//2)

    idxs_upper_triangle = np.array(np.triu_indices(num_modes))

    num_detunings = 0
    num_passive_couplings = 0
    num_active_couplings = 0
    for idx1, idx2, val_coupling in zip(idxs_upper_triangle[0], idxs_upper_triangle[1], arch):
        if val_coupling != NO_COUPLING:
            if idx1 == idx2:
                num_detunings += 1
            elif val_coupling == COUPLING_WITHOUT_PHASE:
                num_passive_couplings += 1
            else:
                num_active_couplings += 1
    return num_detunings, num_passive_couplings, num_active_couplings

def translate_graph_to_conditions(graph):
    num_modes = graph.shape[0]
    conditions = []
    for idx in range(num_modes):
        if graph[idx,idx] == NO_COUPLING:
            conditions.append(Constraint_coupling_zero(idx,idx))
        elif graph[idx,idx] == DETUNING:
            pass
        else:
            raise NotImplementedError()
        
    for idx2 in range(num_modes):
        for idx1 in range(idx2):
            if graph[idx1,idx2] == NO_COUPLING:
                conditions.append(Constraint_coupling_zero(idx1,idx2))
            elif graph[idx1,idx2] == COUPLING_WITHOUT_PHASE:
                conditions.append(Constraint_coupling_phase_zero(idx1,idx2))
            elif graph[idx1,idx2] == COUPLING_WITH_PHASE:
                pass
            else:
                raise NotImplementedError()
    
    return conditions

def translate_upper_triangle_coupling_matrix_to_conditions(coupling_matrix_upper_triangle):
    num_couplings = len(coupling_matrix_upper_triangle)
    num_modes = int((-1 + np.sqrt(1+4*2*num_couplings))//2)

    idxs_upper_triangle = np.array(np.triu_indices(num_modes))
    conditions = []
    for idx1, idx2, val_coupling in zip(idxs_upper_triangle[0], idxs_upper_triangle[1], coupling_matrix_upper_triangle):
        if idx1 == idx2:
            if val_coupling == NO_COUPLING:
                conditions.append(Constraint_coupling_zero(idx1,idx2))
            elif val_coupling == DETUNING:
                pass
            else:
                raise NotImplementedError()
        else:
            if val_coupling == NO_COUPLING:
                conditions.append(Constraint_coupling_zero(idx1,idx2))
            elif val_coupling == COUPLING_WITHOUT_PHASE:
                conditions.append(Constraint_coupling_phase_zero(idx1,idx2))
            elif val_coupling == COUPLING_WITH_PHASE:
                pass
            else:
                raise NotImplementedError()
    
    return conditions

def translate_conditions_to_upper_triangle_coupling_matrix(conditions, num_modes):
    coupling_matrix_upper_triangle = np.zeros((num_modes**2 + num_modes)//2)

    idx_counter = 0
    for idx1, idx2 in np.array(np.triu_indices(num_modes)).T:
        
        if Constraint_coupling_zero(idx1, idx2) in conditions:
            coupling_matrix_upper_triangle[idx_counter] = NO_COUPLING
        elif idx1 != idx2 and Constraint_coupling_phase_zero(idx1, idx2) in conditions:
            coupling_matrix_upper_triangle[idx_counter] = COUPLING_WITHOUT_PHASE
        else:
            if idx1 != idx2:
                coupling_matrix_upper_triangle[idx_counter] = COUPLING_WITH_PHASE
            else:
                coupling_matrix_upper_triangle[idx_counter] = DETUNING
        idx_counter += 1

    return coupling_matrix_upper_triangle


def fill_coupling_matrix(num_modes, detunings, couplings_without_phase, couplings_with_phase):
    detunings_array = np.asarray(detunings)
    couplings_with_phase_array = np.asarray(couplings_with_phase)
    couplings_without_phase_array = np.asarray(couplings_without_phase)

    if len(couplings_with_phase_array) > 0:
        idxs1 = couplings_with_phase_array[:,0]
        idxs2 = couplings_with_phase_array[:,1]
        couplings_with_phase_idxs = (np.concatenate((idxs1, idxs2)), np.concatenate((idxs2, idxs1)))
    else:
        couplings_with_phase_idxs = ([],[])

    if len(couplings_without_phase_array) > 0:
        idxs1 = couplings_without_phase_array[:,0]
        idxs2 = couplings_without_phase_array[:,1]
        couplings_without_phase_idxs = (np.concatenate((idxs1, idxs2)), np.concatenate((idxs2, idxs1)))
    else:
        couplings_without_phase_idxs = ([],[])

    if len(detunings_array) > 0:
        detunings_idxs = (detunings_array, detunings_array)
    else:
        detunings_idxs = ([],[])

    coupling_matrix = np.zeros([num_modes, num_modes], dtype='int')

    coupling_matrix[detunings_idxs] = DETUNING
    coupling_matrix[couplings_with_phase_idxs] = COUPLING_WITH_PHASE
    coupling_matrix[couplings_without_phase_idxs] = COUPLING_WITHOUT_PHASE

    return jnp.array(coupling_matrix)

class Architecture():
    def __init__(self, num_modes=None, detunings=[], couplings_without_phase=[], couplings_with_phase=[], permutation_rules=None, coupling_matrix=None):
        
        if coupling_matrix is not None:
            self.coupling_matrix = coupling_matrix
            self.num_modes = coupling_matrix.shape[0]
        else:
            self.num_modes = num_modes
            self.detunings = detunings
            self.couplings_with_phase = couplings_with_phase
            self.couplings_without_phase = couplings_without_phase
            self.coupling_matrix = fill_coupling_matrix(num_modes, detunings, couplings_without_phase, couplings_with_phase)
        
        self.permutation_rules = permutation_rules
        if permutation_rules is not None:
            raise NotImplementedError()
        
    def is_subgraph_to(self, arch):
        # checks if self (or one of its isomorphic versions) is a subgraph of arch
        # return matrix1_subgraph_to_matrix2(self.coupling_matrix, coupling_matrix)
        return np.sum((self.coupling_matrix - self.coupling_matrix) < 0) == 0


def check_if_subgraph(coupling_matrices, potential_subgraphs):
    if len(coupling_matrices) == 0 or len(potential_subgraphs) == 0:
        return False

    if len(coupling_matrices.shape) > 2 and len(potential_subgraphs.shape) > 2:
        raise NotImplementedError()
    
    return np.any(np.sum((coupling_matrices - potential_subgraphs) < 0, (-1,-2)) == 0)

def check_if_subgraph_upper_triangle(coupling_matrices_upper_triangle, potential_subgraphs_upper_triangle):
    if len(coupling_matrices_upper_triangle) == 0 or len(potential_subgraphs_upper_triangle) == 0:
        return False

    if len(coupling_matrices_upper_triangle.shape) > 1 and len(potential_subgraphs_upper_triangle.shape) > 1:
        raise NotImplementedError()
    
    return np.any(np.sum((coupling_matrices_upper_triangle - potential_subgraphs_upper_triangle) < 0, (-1,)) == 0)

def calc_number_of_possibilities(mode_types, phase_constraints_for_squeezing=False):
    num_modes = len(mode_types)
    num_beamsplitter_couplings = 0
    num_squeezing_couplings = 0
    for idx2 in range(num_modes):
        for idx1 in range(idx2):
            if mode_types[idx1] == mode_types[idx2]:
                num_beamsplitter_couplings += 1
            else:
                num_squeezing_couplings += 1

    return 2**num_modes * 3**num_beamsplitter_couplings * (2+phase_constraints_for_squeezing)**num_squeezing_couplings

    # def __str__(self):
    #     return self.prepare_string_output(False)
    
    # def prepare_string_output(self, extended=False):
    #     if extended:
    #         to_print = self.extended_sets_of_constraints[0]
    #     else:
    #         to_print = self.all_sets_of_constraints[0]
        
    #     output_string = 'Contains the following constraints:\n'
    #     for c in to_print:
    #         output_string += c.__str__() + '\n'
    #     return output_string
    
    # def print(self, extended=False):
    #     print(self.prepare_string_output(extended))

    # def return_characteristics(self):
    #     combo = self.all_sets_of_constraints[0]
    #     num_detuning_constraints = 0
    #     num_coupling_constraints = 0
    #     num_coupling_phase_constraints = 0
    #     for c in combo:
    #         if type(c) == Constraint_coupling_zero:
    #             if c.idxs[0] == c.idxs[1]:
    #                 num_detuning_constraints += 1
    #             else:
    #                 num_coupling_constraints += 1
    #         elif type(c) == Constraint_coupling_phase_zero:
    #             num_coupling_phase_constraints += 1
    #         else:
    #             return NotImplementedError()
        
    #     N = self.num_modes

    #     info = {
    #         'num_detuning_constraints': num_detuning_constraints,
    #         'num_coupling_constraints': num_coupling_constraints,
    #         'num_coupling_phase_constraints': num_coupling_phase_constraints,
    #         'num_constraints': num_detuning_constraints+num_coupling_constraints+num_coupling_phase_constraints,
    #         'num_detunings': N - num_detuning_constraints,
    #         'num_couplings_including_detunings': (N**2-N)//2 + N - num_detuning_constraints - num_coupling_constraints,
    #         'num_couplings_excluding_detunings': (N**2-N)//2 - num_coupling_constraints,
    #     }
        
    #     return info
