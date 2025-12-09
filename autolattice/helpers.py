import numpy as np

def unify_dicts(*dicts):
    keys = dicts[0]
    unified_dict = {}
    for key in keys:
        to_unify = []
        for d in dicts:
            element = d[key]
            if len(element) != 0:
                to_unify.append(element.T)
        unified_dict[key] = np.hstack(to_unify).T

    return unified_dict

def get_solutions_for_graphs(list_of_graphs, unified_dict):
    return np.array([unified_dict['solutions'][np.where(np.all(unified_dict['graphs_tested'] == graph, -1))[0]][0] for graph in list_of_graphs])