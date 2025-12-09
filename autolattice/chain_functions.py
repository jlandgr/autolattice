import autolattice.symbolic as sym

def prepare_operators_chain(mode_types, chain_length):
    modes_per_unit_cell = len(mode_types) - 1
    num_modes = chain_length*modes_per_unit_cell

    mode_types_chain_odd = mode_types[:-1]
    if mode_types[-1] == mode_types[0]:
        mode_types_chain_even = mode_types[:-1]
    else:
        mode_types_chain_even = [not x for x in mode_types[:-1]]

    mode_types_chain = []
    for chain_idx in range(chain_length):
        if chain_idx % 2 == 0:
            mode_types_chain.extend(mode_types_chain_odd)
        else:
            mode_types_chain.extend(mode_types_chain_even)

    operators_chain = []
    for idx in range(num_modes):
        if mode_types_chain[idx]:
            operators_chain.append(sym.Mode().a)
        else:
            operators_chain.append(sym.Mode().adag)

    return operators_chain, mode_types_chain