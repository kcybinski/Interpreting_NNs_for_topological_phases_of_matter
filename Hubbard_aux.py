import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from numba.typed import List, Dict
import scipy as sc
import seaborn as sns
import concurrent.futures
import pathlib
import re
import time


# @njit
def calc_Dim(M, N, statistic="b"):
    """
    A function for calculation of Fock basis dimensions, depending on the
    system's statistic.

    Parameters
    ----------
    M : int
        Number of sites in the 1D chain.
    N : int
        Number of particles in the system.
    statistic : str, optional
        Statistic the system's particles follow. Allowed values are 'f' for
        Fermions, and 'b' for Bosons. The default is "b".

    Raises
    ------
    ValueError
        This error is raised if incorrect statistic alias is entered.

    Returns
    -------
    dim : int
        Fock basis dimension.

    """
    if statistic == "b":
        return int(np.math.gamma(N + M) / (np.math.gamma(N + 1) * np.math.gamma(M)))
    elif statistic == "f":
        # return np.int64(
        #     (np.math.gamma(M + 1)) / (np.math.gamma(N + 1) * np.math.gamma(M - N + 1))
        # )
        return np.int64(sc.special.comb(M, N, exact=True))
    else:
        raise ValueError("Incorrect statistic")


# @njit
def calc_dim_tab(M, N_tab, stat_tab):
    dim_arr = np.array([1.0], dtype=np.float64)
    for (n_i, stat_i) in zip(N_tab, stat_tab):
        if stat_i == "b":
            D = calc_Dim(M, n_i, stat_i)
            dim_arr = np.append(dim_arr, [D])
        elif stat_i == "f":
            if n_i > M:
                raise ValueError("Non-physical fermion number")
            dim_F = calc_Dim(M, n_i, stat_i)
            dim_arr = np.append(dim_arr, [dim_F])
        else:
            raise ValueError("Incorrect statistic")
    return int(dim_arr.prod())


# def show_wave_function(
#     psket, Fock_Basis, high_num=3, plot_wf=False, label_ticks=False, print_all=False
# ):
#     res_dict = {}
#     ket_to_sort = psket.copy()
#     for alpha, basis_vec in zip(psket, Fock_Basis):
#         res_dict[alpha] = basis_vec
#     ket_to_sort[::-1].sort()
#     highest_prob = ket_to_sort[0:high_num]
#     for el in highest_prob:
#         print("{:.4e}".format(el**2), res_dict[el])
#     if print_all:
#         for alpha, basis_vec in zip(psket, Fock_Basis):
#             print("{:.4e}".format(alpha**2), basis_vec)
#     if plot_wf:
#         fig, ax = plt.subplots(1, 1, figsize=(10, 8))
#         if label_ticks:
#             ax.set_xticks(np.arange(0, Fock_Basis.shape[0], 1))
#             ax.set_xticklabels(Fock_Basis, rotation=90)
#         ax = plt.plot(
#             np.arange(0, Fock_Basis.shape[0], 1),
#             [(lambda alpha: alpha**2)(alpha) for alpha in psket],
#         )
#         # ax.bar(np.arange(0, Fock_Basis.shape[0], 1), [(lambda alpha: alpha**2)(alpha) for alpha in psket])

#         # ticker.StrMethodFormatter('{x}')
#         # ax.axis.se


def show_wave_function(
    psket,
    Fock_Basis,
    high_num=3,
    plot_wf=False,
    label_ticks=False,
    print_all=False,
    bar=False,
    plot_prob=False,
    k=None,
    ax_apriori=None,
):
    res_dict = {}
    ket_to_sort = psket.copy()
    for alpha, basis_vec in zip(psket, Fock_Basis):
        res_dict[alpha] = basis_vec
    ket_to_sort[::-1].sort()
    highest_prob = ket_to_sort[0:high_num]
    for el in highest_prob:
        print("{:.4e}".format(el**2), res_dict[el])
    if print_all:
        for alpha, basis_vec in zip(psket, Fock_Basis):
            print("{:.4e}".format(alpha**2), basis_vec)
    if plot_wf:
        if ax_apriori is None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        else:
            ax = ax_apriori
        if k is not None:
            ax.set_title("L = {} | k = {}".format(psket.shape[0], k))
        if label_ticks:
            ax.set_xticks(np.arange(0, Fock_Basis.shape[0], 1))
            ax.set_xticklabels(Fock_Basis, rotation=90)
        if plot_prob:
            if bar:
                ax.bar(
                    np.arange(0, Fock_Basis.shape[0], 1),
                    [(lambda alpha: alpha**2)(alpha) for alpha in psket],
                )
            else:
                ax.plot(
                    np.arange(0, Fock_Basis.shape[0], 1),
                    [(lambda alpha: alpha**2)(alpha) for alpha in psket],
                )
        else:
            if bar:
                ax.bar(np.arange(0, Fock_Basis.shape[0], 1), psket)
            else:
                ax.plot(np.arange(0, Fock_Basis.shape[0], 1), psket)


# @njit(parallel=True)
def get_basis(M, N, statistic="b", verb=0):
    """
    A function generating a Fock Basis for many-body calculations. Default is for Bosons, optional for Fermions

    :param M:
    :param N:
    :param verb:
    :param statistic:
    :return:
    """
    # Defining empty array for the basis vectors
    if statistic == "b":
        D = calc_Dim(M, N, "b")
        A = np.zeros(shape=(D, M), dtype=np.int64)
    elif statistic == "f":
        if N > M:
            raise ValueError("Non-physical fermion number")
        D = calc_Dim(M, N, "b")
        A = np.zeros(shape=(D, M), dtype=np.int64)
        # dim_F = math.comb(M, N)
        dim_F = calc_Dim(M, N, statistic)
        A_F = np.zeros(shape=(dim_F, M), dtype=np.int64)
        a_f_ind = 0
    else:
        raise ValueError("Incorrect statistic")

    # Setting up the basis vectors
    for num, row in enumerate(A):
        if num != 0:
            row_old = A[num - 1].copy()
            row_new = row.copy()
            k_ind = np.max(np.nonzero(row_old[: M - 1])[0])
            if row_old[k_ind + 1] != 0:
                k_ind = np.max(np.nonzero(A[num - 2][: M - 1])[0])
            # Setting the new row
            for i in range(0, k_ind):
                row_new[i] = row_old[i]
            row_new[k_ind] = row_old[k_ind] - 1
            existing_sum = 0
            for pos in range(len(row_new)):
                if pos >= k_ind + 3:
                    continue
                else:
                    existing_sum += row_new[pos]
            row_new[k_ind + 1] = N - existing_sum

            # Copying back the new row
            A[num] = row_new
            if statistic == "f":
                if np.max(row_new) < 2:
                    A_F[a_f_ind] = row_new
                    a_f_ind += 1
        else:
            row[0] = N
            if statistic == "f":
                if np.max(row) < 2:
                    A_F[a_f_ind] = row
                    a_f_ind += 1

    if verb > 0:
        if statistic == "b" or statistic == "f_2":
            print(A)
        elif statistic == "f":
            print(A_F)
        else:
            raise ValueError("Incorrect statistic")
    if statistic == "b":
        A[-1, -1] = N
        return A
    elif statistic == "f":
        A_F[-1, -1] = 1
        return A_F
    else:
        raise ValueError("Incorrect statistic")


# @njit
def get_tensor_basis(M, N, statistic=["b"], component_count=1, verb=0):
    """
    A function for generation of Fock basis for 1 or 2 component systems of
    either Fermi-Dirac or Bose-Einstein statistic.

    Parameters
    ----------
    M : int
        Number of sites in 1D chain.
    N : np.array
        An array containing particle counts in either of the components.
        Its length should match the component_count parameter.
    statistic : np.array, optional
        An array containing particle statistics in either of the components.
        Possible values are 'f' for fermions, and 'b' for bosons.
        Its length should match the component_count parameter.
        The default is ["b"].
    component_count : int, optional
        Number of Fock basis components. For composite/spinful systems it
        should be 2, for simple ones, it should be 1. The default is 1.
    verb : int, optional
        If verb > 0m then technical prints will show up. The default is 0.

    Raises
    ------
    ValueError
        Raises this error if 'N' is not a NumPy array, or if incorrect
        statistic is passed.

    Returns
    -------
    A : np.array
        An array containing Fock basis vectors as rows sorted lexicographically.

    """
    if len(N) == 1:
        N_val = N[0]
        stat_val = statistic[0]
        vec_num = 0
        if stat_val == "b":
            D = calc_Dim(M, N_val, "b")
            A = np.zeros(shape=(D**2, M * component_count), dtype=np.int64)
        elif stat_val == "f":
            if N_val > M:
                raise ValueError("Non-physical fermion number")
            dim_F = calc_Dim(M, N_val, stat_val)
            A = np.zeros(shape=(dim_F**2, M * component_count), dtype=np.int64)
        else:
            raise ValueError("Incorrect statistic")
        if component_count == 1:
            return get_basis(M, N_val, statistic=stat_val, verb=verb)
        else:
            bases = {}
            for comp_num in range(0, component_count):
                if verb > 0:
                    print("Generating basis for component no.", comp_num)
                bases[comp_num] = get_basis(M, N_val, statistic=stat_val, verb=verb)

            for ket_1 in bases[0]:
                for ket_2 in bases[1]:
                    A[vec_num] = np.concatenate((ket_1, ket_2))
                    vec_num += 1
        return A
    elif len(N) > 1:
        N_tab = N
        stat_tab = statistic

        dim_prod = calc_dim_tab(M, N_tab, stat_tab)
        A = np.zeros(shape=(dim_prod, M * component_count), dtype=np.int64)

        vec_num = 0
        if component_count == 1:
            return get_basis(M, N_tab[0], statistic=statistic, verb=verb)
        else:
            bases = {}
            for (n_i, stat_i, comp_num) in zip(N, stat_tab, range(0, component_count)):
                if verb > 0:
                    print("Generating basis for component no.", comp_num)
                bases[comp_num] = get_basis(M, n_i, statistic=stat_i, verb=verb)

            for ket_1 in bases[0]:
                for ket_2 in bases[1]:
                    A[vec_num] = np.concatenate((ket_1, ket_2))
                    vec_num += 1
        return A
    else:
        raise ValueError("Please pass 'N' as a list")


@njit
def a_operator(
    i_ind,
    ket,
    kind="c",
    component_number=None,
    ket_size=None,
    statistic="b",
    pbc=0,
    component_count=1,
):
    """
    A joint function for anihillation and creation operators for bosons (b_i, b_i^\dagger) and fermions (c_i, c_i^\dagger) which acts on input ket from multi-particle Fock basis.

    :param i_ind: Index of interest (as in a_i)
    :param ket: The basis vector the operator acts on
    :param kind: Operator kind, either 'c' for creation' or 'a' for anihillation is permitted
    :param component_number: Number of component in a tensor-multiplied basis. If none, then singular basis is assumed. Numbering from 0
    :param ket_size: Size of single component of tensor-multiplied basis
    :param statistic: Particle statistic, 'b' for Boson, or 'f' for Fermion
    :param pbc: Periodic Boundary Conditions. 0 - off, 1 - on
    :return: Returns multiplying factor, and new ket
    """
    # Setting which site should be modified based on boundary conditions.
    if pbc == 0:
        i_mod = i_ind
    else:
        i_mod = np.mod(i_ind, ket.shape[0])
    # Creation operator
    if kind == "c":
        if statistic == "b":
            if component_count == 1:
                factor = np.sqrt(ket[i_mod] + 1)
                ket_new = ket.copy()
                ket_new[i_mod] += 1
            else:
                ket_to_mod = ket[
                    component_number * ket_size : (component_number + 1) * ket_size
                ]
                factor = np.sqrt(ket_to_mod[i_mod] + 1)
                ket_new = ket_to_mod.copy()
                ket_new[i_mod] += 1
                before = ket[0 : (component_number) * ket_size]
                after = ket[
                    (component_number + 1)
                    * ket_size : (component_number + 2)
                    * ket_size
                ]
                ket_out = np.concatenate((before, ket_new, after))
                return factor, ket_out
        elif statistic == "f":
            if component_count == 1:
                if ket[i_mod] == 1:
                    return 0, ket
                factor = np.sqrt(ket[i_mod] + 1)
                ket_new = ket.copy()
                ket_new[i_mod] += 1
            else:
                ket_to_mod = ket[
                    component_number * ket_size : (component_number + 1) * ket_size
                ]
                if ket_to_mod[i_mod] == 1:
                    return 0, ket
                minus_factor = np.count_nonzero(
                    ket[
                        component_number * ket_size : component_number * ket_size
                        + i_mod
                    ]
                )
                factor = np.sqrt(ket_to_mod[i_mod] + 1) * np.power(-1, minus_factor)
                ket_new = ket_to_mod.copy()
                ket_new[i_mod] += 1
                # print(ket_new)
                before = ket[0 : (component_number) * ket_size]
                after = ket[
                    (component_number + 1)
                    * ket_size : (component_number + 2)
                    * ket_size
                ]
                ket_out = np.concatenate((before, ket_new, after))
                # print(ket_out)
                return factor, ket_out
    # Anihillation operator
    if kind == "a":
        if statistic == "b":
            if component_count == 1:
                if ket[i_mod] == 0:
                    return 0, ket
                factor = np.sqrt(ket[i_mod])
                ket_new = ket.copy()
                ket_new[i_mod] -= 1
            else:
                ket_to_mod = ket[
                    component_number * ket_size : (component_number + 1) * ket_size
                ]
                if ket_to_mod[i_mod] == 0:
                    return 0, ket
                factor = np.sqrt(ket_to_mod[i_mod])
                ket_new = ket_to_mod.copy()
                ket_new[i_mod] -= 1
                before = ket[0 : (component_number) * ket_size]
                after = ket[
                    (component_number + 1)
                    * ket_size : (component_number + 2)
                    * ket_size
                ]
                ket_out = np.concatenate((before, ket_new, after))
                return factor, ket_out
        elif statistic == "f":
            if component_count == 1:
                if ket[i_mod] == 0:
                    return 0, ket
                factor = np.sqrt(ket[i_mod])
                ket_new = ket.copy()
                ket_new[i_mod] -= 1
            else:
                ket_to_mod = ket[
                    component_number * ket_size : (component_number + 1) * ket_size
                ]
                if ket_to_mod[i_mod] == 0:
                    return 0, ket
                minus_factor = np.count_nonzero(
                    ket[
                        component_number * ket_size : component_number * ket_size
                        + i_mod
                    ]
                )
                factor = np.sqrt(ket_to_mod[i_mod]) * np.power(-1, minus_factor)
                ket_new = ket_to_mod.copy()
                ket_new[i_mod] -= 1
                # print(ket_new)
                before = ket[0 : (component_number) * ket_size]
                after = ket[
                    (component_number + 1)
                    * ket_size : (component_number + 2)
                    * ket_size
                ]
                ket_out = np.concatenate((before, ket_new, after))
                # print(ket_out)
                return factor, ket_out
    return factor, ket_new


def calc_IPR(ket):
    return 1 / np.sum([(lambda i: ket[i] ** 4)(i) for i in range(0, ket.shape[0])])


@njit
def tag_func(v):
    """
    The tagging function, which hashes all the basis vectors with unique values, according to equation:
    T(v) = \sum_i \sqrt{p_i} v_i where i is the index of element in the vector

    :param v: A vector to hash
    :return: Hash of the vector
    """
    res_tab = np.zeros(len(v))
    for i in range(0, len(v)):
        res_tab[i] = np.sqrt(100 * (i + 1) + 3) * v[i]
    return np.sum(res_tab)


@njit
def find_orig_idx(ket, t_dict):
    """
    Function for finding which basis vector is the input ket

    :param ket: Hashed ket to be searched in the original Hilbert space basis
    :param t_dict: dictionary of hashed basis vectors
    :return: Index of input vector in the original ordered basis of the Hilbert space
    """
    ket_hash = tag_func(ket)
    try:
        a = t_dict[ket_hash]
        return int(a)
    except Exception:
        return -1
    # sorted_idx = np.searchsorted(t_sorted, ket_hash)
    # if ket_hash != t_sorted[sorted_idx]:
    #     # Here taking care of vectors that are not in the basis
    #     return -1
    # else:
    #     return ind[sorted_idx]


@njit
def get_kinetic_H(
    A, M, J, t_dict, delta_t=0, pbc=0, statistic="b", component_no=1, component_count=1
):
    values = List()

    ket_idx_list = List()

    ket_tilde_tilde_idx_list = List()

    ket_len = M

    if pbc == 0:
        range_end = M - 1
    else:
        range_end = M

    # for component_no in range(0, component_count):
    for ket in A:
        for i in range(0, range_end):

            if pbc == 0:
                i_next = i + 1
            else:
                i_next = np.mod(i + 1, ket_len)

            for (anihillated, created) in [(i, i_next), (i_next, i)]:
                if component_count > 1:
                    multicomp_factor = component_no * ket_len
                else:
                    multicomp_factor = 0
                if ket[multicomp_factor + anihillated] > 0:

                    factor_1, ket_tilde = a_operator(
                        anihillated,
                        ket,
                        "a",
                        pbc=pbc,
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_2, ket_tilde_tilde = a_operator(
                        created,
                        ket_tilde,
                        "c",
                        pbc=pbc,
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )

                    ket_idx = find_orig_idx(ket, t_dict)
                    ket_tilde_tilde_idx = find_orig_idx(ket_tilde_tilde, t_dict)

                    val = -(1 + delta_t * ((-1) ** (i_next))) * factor_1 * factor_2 * J

                    if ket_tilde_tilde_idx == -1 or np.abs(val) == 0:
                        continue
                    else:

                        values.append(val)

                        ket_idx_list.append(ket_idx)

                        ket_tilde_tilde_idx_list.append(ket_tilde_tilde_idx)

    return values, ket_idx_list, ket_tilde_tilde_idx_list


def get_prob_mat(evecs):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    mat_to_plot = np.square(abs(evecs))
    sns.heatmap(mat_to_plot, cmap="coolwarm")
    return fig


def get_kinetic_H_vw(
    A,
    M,
    J,
    t_dict,
    vw=[1, 1],
    disord_W=[0, 0],
    pbc=0,
    statistic="b",
    component_no=1,
    component_count=1,
):
    """
    A function calculating the kinetic Hamiltonian of a SSH system, defined with staggered hopping amplitudes v, w.

    Parameters
    ----------
    A : Numpy Array of int64
        Fock Basis on which we operate, either fermionic, or bosonic. Base's statistic should match the one given in 'statistic' parameter.
    M : int
        Number of sites in the system. A site is a building block of elementary cell.
    J : int
        Hopping scaling factor, as stated in Hamiltonian definition.
    t_dict : numba.typed.typedict.Dict
        A dictionary connecting hashed and sorted basis vectors with their original positions.
    vw : Numpy Array of float64, optional
        A vector containing hopping amplitudes v, w. The default is [1, 1].
    disord_W : Numpy Array of float64, optional
        An array setting the disorder amplitude on respectively inter and intra cell hopping. The default is [0, 0].
    pbc : int, optional
        Periodic (1) or open(0) boundary conditions imposed on SSH chain. The default is 0.
    statistic : Numpy Array of str32, optional
        Statistic describing the particles in the system 'b' - bosonic, 'f' - fermionic. The default is "b".
    component_no : int, optional
        Which part of Fock basis vector should be acted on. Only matters if presented with two-component Fock basis. The default is 1.
    component_count : int, optional
        Declaration of Fock basis components in the system. The default is 1.

    Returns
    -------
    values : numba.typed.typedlist.List
        A list of values to assign to sparse matrix - Hamiltonian.
    ket_idx_list : numba.typed.typedlist.List
        An row number list in the resulting sparse matrix, corresponding to the indices of original basis vectors.
    ket_tilde_tilde_idx_list : numba.typed.typedlist.List
        An column number list in the resulting sparse matrix, corresponding to the indices of original basis vectors.

    """

    values = List()

    ket_idx_list = List()

    ket_tilde_tilde_idx_list = List()

    ket_len = M

    if pbc == 0:
        range_end = M - 1
    else:
        range_end = M

    for ket in A:
        # We set independent random generator states for genrator
        # Seed we set is system time modulo max integer capacity
        rs = np.random.RandomState(time.time_ns() % (2**32 - 1))
        r_1 = rs.uniform(-0.5, 0.5)
        W_1_disord = disord_W[0] * r_1
        rs = np.random.RandomState(time.time_ns() % (2**32 - 1))
        r_2 = rs.uniform(-0.5, 0.5)
        W_2_disord = disord_W[1] * r_2

        # Hitchhiker's guide through hopping:
        # w + W_1 * r_n' <- Intercell Hopping (Between neighboring cells)
        # v + W_2 * r_n  <- Intracell Hopping (Within unit cell)

        stag_dict = {1: (vw[1] + W_1_disord), -1: (vw[0] + W_2_disord), 0: 0}

        for i in range(0, range_end):

            if pbc == 0:
                i_next = i + 1
            else:
                i_next = np.mod(i + 1, ket_len)

            for (anihillated, created) in [(i, i_next), (i_next, i)]:
                if component_count > 1:
                    multicomp_factor = component_no * ket_len
                else:
                    multicomp_factor = 0
                if ket[multicomp_factor + anihillated] > 0:

                    factor_1, ket_tilde = a_operator(
                        anihillated,
                        ket,
                        "a",
                        pbc=pbc,
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_2, ket_tilde_tilde = a_operator(
                        created,
                        ket_tilde,
                        "c",
                        pbc=pbc,
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )

                    ket_idx = find_orig_idx(ket, t_dict)
                    ket_tilde_tilde_idx = find_orig_idx(ket_tilde_tilde, t_dict)

                    val = (stag_dict[((-1) ** (i_next))]) * factor_1 * factor_2 * J

                    if ket_tilde_tilde_idx == -1 or np.abs(val) == 0:
                        continue
                    else:

                        values.append(val)

                        ket_idx_list.append(ket_idx)

                        ket_tilde_tilde_idx_list.append(ket_tilde_tilde_idx)

    return values, ket_idx_list, ket_tilde_tilde_idx_list


def get_kinetic_H_vw_NNN(
    A,
    M,
    J,
    t_dict,
    tvar=1,
    disord_W=[0],
    pbc=0,
    statistic="b",
    component_no=1,
    component_count=1,
):

    values = List()

    ket_idx_list = List()

    ket_tilde_tilde_idx_list = List()

    ket_len = M

    if pbc == 0:
        range_end = M - 2
    else:
        range_end = M

    for ket in A:
        # todo - set system time seed
        rs = np.random.RandomState(time.time_ns() % (2**32 - 1))
        W_disord = disord_W[0] * rs.uniform(-0.5, 0.5)

        stag_dict = {-1: (tvar + W_disord), 0: 0, 1: (tvar + W_disord)}

        for i in range(0, range_end):

            if pbc == 0:
                i_next = i + 3
            else:
                i_next = np.mod(i + 3, ket_len)

            for (anihillated, created) in [(i, i_next), (i_next, i)]:
                if component_count > 1:
                    multicomp_factor = component_no * ket_len
                else:
                    multicomp_factor = 0
                if ket[multicomp_factor + anihillated] > 0:

                    factor_1, ket_tilde = a_operator(
                        anihillated,
                        ket,
                        "a",
                        pbc=pbc,
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_2, ket_tilde_tilde = a_operator(
                        created,
                        ket_tilde,
                        "c",
                        pbc=pbc,
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )

                    ket_idx = find_orig_idx(ket, t_dict)
                    ket_tilde_tilde_idx = find_orig_idx(ket_tilde_tilde, t_dict)

                    val = (stag_dict[((-1) ** (i_next))]) * factor_1 * factor_2 * J

                    if ket_tilde_tilde_idx == -1 or np.abs(val) == 0:
                        continue
                    else:

                        values.append(val)

                        ket_idx_list.append(ket_idx)

                        ket_tilde_tilde_idx_list.append(ket_tilde_tilde_idx)

    return values, ket_idx_list, ket_tilde_tilde_idx_list


# @njit
def get_interactions_H(
    A, M, V, t_dict, statistic="b", component_no=1, component_count=1, pbc=0
):
    values = List()

    ket_idx_list = List()

    ket_tilde_tilde_idx_list = List()

    ket_len = M

    if statistic == "b" and V != 0:
        for ket in A:
            for i in range(0, M):
                if component_count > 1:
                    multicomp_factor = component_no * ket_len
                else:
                    multicomp_factor = 0
                if ket[multicomp_factor + i] > 0:
                    factor_1, ket_t1 = a_operator(
                        i,
                        ket,
                        "a",
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_2, ket_t2 = a_operator(
                        i,
                        ket_t1,
                        "c",
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_3, ket_t3 = a_operator(
                        i,
                        ket_t2,
                        "a",
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_4, ket_t4 = a_operator(
                        i,
                        ket_t3,
                        "c",
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )

                    ket_idx = find_orig_idx(ket, t_dict)
                    ket_tilde_tilde_idx = find_orig_idx(ket_t4, t_dict)

                    val = (V / 2.0) * factor_1 * factor_2 * factor_3 * factor_4

                    values.append(val)

                    ket_idx_list.append(ket_idx)

                    ket_tilde_tilde_idx_list.append(ket_tilde_tilde_idx)

                    # Second part of the equation

                    factor_1_bis, ket_t1_bis = a_operator(
                        i,
                        ket,
                        "a",
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_2_bis, ket_t2_bis = a_operator(
                        i,
                        ket_t1_bis,
                        "c",
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )

                    ket_idx = find_orig_idx(ket, t_dict)
                    ket_tilde_tilde_idx_bis = find_orig_idx(ket_t2_bis, t_dict)

                    val_bis = (-1) * (V / 2.0) * factor_1_bis * factor_2_bis

                    values.append(val_bis)

                    ket_idx_list.append(ket_idx)

                    ket_tilde_tilde_idx_list.append(ket_tilde_tilde_idx_bis)
    elif statistic == "f" and V != 0:
        for ket in A:
            if pbc == 0:
                range_end = M - 1
            else:
                range_end = M

            for i in range(0, range_end):
                if component_count > 1:
                    multicomp_factor = component_no * ket_len
                else:
                    multicomp_factor = 0
                if pbc == 0:
                    i_next = i + 1
                else:
                    i_next = np.mod(i + 1, ket_len)

                if ket[multicomp_factor + i] > 0 and ket[multicomp_factor + i_next] > 0:
                    factor_1, ket_t1 = a_operator(
                        i,
                        ket,
                        "a",
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_2, ket_t2 = a_operator(
                        i,
                        ket_t1,
                        "c",
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_3, ket_t3 = a_operator(
                        i_next,
                        ket_t2,
                        "a",
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_4, ket_t4 = a_operator(
                        i_next,
                        ket_t3,
                        "c",
                        statistic=statistic,
                        component_number=component_no,
                        ket_size=ket_len,
                        component_count=component_count,
                    )

                    ket_idx = find_orig_idx(ket, t_dict)
                    ket_tilde_tilde_idx = find_orig_idx(ket_t4, t_dict)

                    val = (V / 2.0) * factor_1 * factor_2 * factor_3 * factor_4

                    values.append(val)

                    ket_idx_list.append(ket_idx)

                    ket_tilde_tilde_idx_list.append(ket_tilde_tilde_idx)

    else:
        pass

    return values, ket_idx_list, ket_tilde_tilde_idx_list


@njit
def get_intercomp_interactions_H(
    A, M, U, t_dict, statistic=["b", "b"], component_count=2
):
    values = List()

    ket_idx_list = List()

    ket_tilde_tilde_idx_list = List()

    ket_len = M

    if component_count < 2:
        raise ValueError("Too little components!")

    if U != 0:
        for ket in A:
            for i in range(0, M):
                comp_1 = 0
                comp_2 = 1

                if ket[comp_1 * ket_len + i] > 0 and ket[comp_2 * ket_len + i] > 0:
                    factor_1, ket_t1 = a_operator(
                        i,
                        ket,
                        "a",
                        statistic=statistic[comp_2],
                        component_number=comp_2,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_2, ket_t2 = a_operator(
                        i,
                        ket_t1,
                        "c",
                        statistic=statistic[comp_2],
                        component_number=comp_2,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_3, ket_t3 = a_operator(
                        i,
                        ket_t2,
                        "a",
                        statistic=statistic[comp_1],
                        component_number=comp_1,
                        ket_size=ket_len,
                        component_count=component_count,
                    )
                    factor_4, ket_t4 = a_operator(
                        i,
                        ket_t3,
                        "c",
                        statistic=statistic[comp_1],
                        component_number=comp_1,
                        ket_size=ket_len,
                        component_count=component_count,
                    )

                    ket_idx = find_orig_idx(ket, t_dict)
                    ket_tilde_tilde_idx = find_orig_idx(ket_t4, t_dict)

                    val = (U) * factor_1 * factor_2 * factor_3 * factor_4

                    values.append(val)

                    ket_idx_list.append(ket_idx)

                    ket_tilde_tilde_idx_list.append(ket_tilde_tilde_idx)

    else:
        pass

    return values, ket_idx_list, ket_tilde_tilde_idx_list


def plot_coo_matrix(m, title=None):
    """
    A function for plotting sparse matrix sparcity pattern

    :param m: Matrix to be plotted
    :param title: Optional variable for labeling the plot
    :return: Matplotlib axis object
    """
    if not isinstance(m, sc.sparse.coo_matrix):
        m = sc.sparse.coo_matrix(m)
    fig = plt.figure(figsize=(8, 8), facecolor="silver")
    ax = fig.add_subplot(111)
    ax.plot(m.col, m.row, "s", color="k", ms=2)
    ax.set_xlim(0, m.shape[1])
    ax.set_ylim(0, m.shape[0])
    ax.set_aspect("equal")
    if title:
        ax.set_title(title, fontsize=20, color="firebrick")
    for spine in ax.spines.values():
        spine.set_visible(True)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    # ax.set_xticks([])
    # ax.set_yticks([])
    return ax


def save_sparse_coo(filename, array):
    # note that .npz extension is added automatically
    sc.sparse.save_npz(filename, array)


def load_sparse_coo(filename):
    # here we need to add .npz extension manually
    loader = sc.sparse.load_npz(filename + ".npz")
    return loader


def get_Hamiltonian(
    N,
    M,
    J,
    V,
    pbc=0,
    statistic=["b"],
    delta_t=0,
    component_count=1,
    component_no=0,
    basis_apriori=None,
    save_sparse=False,
    plot_accurate=False,
    annot_accurate=False,
    plot_sparsity_pattern=False,
    save_plot=False,
    return_basis=False,
    plot_show=False,
):
    """
    A function for generating a 1-D Bose-Hubbard model Hamiltonian. The system is defined by parameters N - total number of particles in the system, M - total number of sites in the chain, J - Hopping parameter, V - On-site potential parameter.

    :param N: Total number of paricles in the system
    :param M: Total number of sites in the chain
    :param J: Hoppin parameter
    :param V: On-site potential parameter
    :param pbc: Periodic Boundary Conditions. 0 - off, 1 - on
    :param save_sparse: Save resulting Hamitonian in sparse format in the working directory
    :param plot_accurate: Plot accurately all the fields of the resulting Hamiltonian. If 'annot_accurate' kwarg is set to True, then actual values are also annotated
    :param annot_accurate: Only takes effect if `plot_accurate` is set to True, annotates plotted Hamiltonian with exact values
    :param plot_sparsity_pattern: Plots general shape of non-zero element in Hamiltonian. Vseful with large matrices, which are not suitable for exact plotting
    :param save_plot: Saves plotted precise plot/sparcity pattern
    :param return_basis: If set to True, then the generated Fock basis is also returned
    :param plot_show: Should drawn plot be shown, flag added for just saving generated plots
    :return: Hamiltonian in COO sparse matrix format, or a tuple containing spare Hamiltonian, and a dense matrix with Fock basis states
    """
    D = calc_dim_tab(M, N, statistic)

    if basis_apriori is None:
        # Setting the occupational basis:
        A = get_tensor_basis(M, N, statistic=statistic, component_count=component_count)
    else:
        A = basis_apriori

    # Getting the basis vectors hashed
    tab_T = np.array([tag_func(v) for v in A])
    # # Preserving the original order of basis vectors
    # ind = np.argsort(tab_T)
    # # Sorting the new array for more efficient searching (by bisection)
    # t_sorted = tab_T.copy()
    # t_sorted.sort()
    t_dict = Dict()
    for key, val in zip(tab_T, np.arange(0, A.shape[0])):
        t_dict[key] = val

    dim_mat = (D, D)

    for el_no in range(0, component_count):
        # First tab - values, last two tabs - coordinates
        H_kin_prep = [List(), List(), List()]
        H_int_prep = [List(), List(), List()]

        H_kin_prep = get_kinetic_H(
            A,
            M,
            J,
            t_dict,
            delta_t=delta_t,
            pbc=pbc,
            statistic=statistic[el_no],
            component_no=el_no,
            component_count=component_count,
        )
        H_int_prep = get_interactions_H(
            A,
            M,
            V[el_no],
            t_dict,
            statistic=statistic[el_no],
            component_no=el_no,
            component_count=component_count,
        )

        mat = sc.sparse.coo_matrix(
            (H_kin_prep[0], (H_kin_prep[1], H_kin_prep[2])), shape=dim_mat
        )
        mat_int = sc.sparse.coo_matrix(
            (H_int_prep[0], (H_int_prep[1], H_int_prep[2])), shape=dim_mat
        )

        if el_no == 0:
            Total_H = mat + mat_int
        else:
            Total_H += mat + mat_int

    if save_sparse:
        p = (
            pathlib.Path.cwd()
            .joinpath("{}_component".format(component_count))
            .joinpath("Hamiltonians")
            .joinpath("N={}_L={}".format(N, M))
        )
        if not p.exists():
            p.mkdir(parents=True, exist_ok=True)
        sc.sparse.save_npz(
            "{}_component/Hamiltonians/N={}_L={}/{}_Hamiltonian_N={}_L={}_V_J=".format(
                component_count,
                N,
                M,
                "".join(
                    [(lambda el: "B" if el == "b" else "F")(el) for el in statistic]
                )
                + "H",
                N,
                M,
            )
            + "_".join(
                [
                    (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                    for el in np.abs(V / J)
                ]
            )
            + "_{}".format("OBC" if pbc == 0 else "PBC"),
            Total_H,
        )

    if plot_sparsity_pattern:
        ax = plot_coo_matrix(
            Total_H,
            title="Stat = "
            + ", ".join([(lambda el: "B" if el == "b" else "F")(el) for el in stat])
            + "; N = "
            + ", ".join([(lambda el: "{}".format(el))(el) for el in N])
            + "; L = {}; V/J = ".format(M)
            + ", ".join(
                [
                    (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                    for el in np.abs(V / J)
                ]
            ),
        )
        if save_plot:
            p_sparse = (
                pathlib.Path.cwd()
                .joinpath("{}_component".format(component_count))
                .joinpath("Plots")
                .joinpath("Hamiltonians")
                .joinpath("N={}_L={}".format(N, M))
            )
            if not p_sparse.exists():
                p_sparse.mkdir(parents=True, exist_ok=True)
            if pbc == 1:
                ax.figure.savefig(
                    "{}_component/Plots/Hamiltonians/N={}_L={}/{}_Hamiltonian_sparse_N={}_L={}_V_J=".format(
                        component_count,
                        N,
                        M,
                        "".join(
                            [
                                (lambda el: "B" if el == "b" else "F")(el)
                                for el in statistic
                            ]
                        )
                        + "H",
                        N,
                        M,
                    )
                    + "_".join(
                        [
                            (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                            for el in np.abs(V / J)
                        ]
                    )
                    + "_{}.pdf".format("OBC" if pbc == 0 else "PBC")
                )
            else:
                ax.figure.savefig(
                    "{}_component/Plots/Hamiltonians/N={}_L={}/{}_Hamiltonian_sparse_N={}_L={}_V_J=".format(
                        component_count,
                        N,
                        M,
                        "".join(
                            [
                                (lambda el: "B" if el == "b" else "F")(el)
                                for el in statistic
                            ]
                        )
                        + "H",
                        N,
                        M,
                    )
                    + "_".join(
                        [
                            (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                            for el in np.abs(V / J)
                        ]
                    )
                    + "_{}.pdf".format("OBC" if pbc == 0 else "PBC")
                )
        if plot_show:
            ax.figure.show()
        plt.close()

    if plot_accurate:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.set_title(
            "Stat = "
            + ", ".join([(lambda el: "B" if el == "b" else "F")(el) for el in stat])
            + "; N = "
            + ", ".join([(lambda el: "{}".format(el))(el) for el in N])
            + "; L = {}; V/J = ".format(M)
            + ", ".join(
                [
                    (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                    for el in np.abs(V / J)
                ]
            )
        )
        sns.heatmap(
            Total_H.toarray(), square=True, cmap="coolwarm", annot=annot_accurate, ax=ax
        )
        if save_plot:
            p_plot = (
                pathlib.Path.cwd()
                .joinpath("{}_component".format(component_count))
                .joinpath("Plots")
                .joinpath("Hamiltonians")
                .joinpath("N={}_L={}".format(N, M))
            )
            if not p_plot.exists():
                p_plot.mkdir(parents=True, exist_ok=True)
            if pbc == 1:
                fig.savefig(
                    "{}_component/Plots/Hamiltonians/N={}_L={}/{}_Hamiltonian_N={}_L={}_V_J=".format(
                        component_count,
                        N,
                        M,
                        "".join(
                            [
                                (lambda el: "B" if el == "b" else "F")(el)
                                for el in statistic
                            ]
                        )
                        + "H",
                        N,
                        M,
                    )
                    + "_".join(
                        [
                            (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                            for el in np.abs(V / J)
                        ]
                    )
                    + "_{}.pdf".format("OBC" if pbc == 0 else "PBC")
                )
            else:
                fig.savefig(
                    "{}_component/Plots/Hamiltonians/N={}_L={}/{}_Hamiltonian_N={}_L={}_V_J=".format(
                        component_count,
                        N,
                        M,
                        "".join(
                            [
                                (lambda el: "B" if el == "b" else "F")(el)
                                for el in statistic
                            ]
                        )
                        + "H",
                        N,
                        M,
                    )
                    + "_".join(
                        [
                            (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                            for el in np.abs(V / J)
                        ]
                    )
                    + "_{}.pdf".format("OBC" if pbc == 0 else "PBC")
                )
        if plot_show:
            plt.show(fig)
        plt.close()

    if return_basis:
        return Total_H, A
    else:
        return Total_H


@njit
def fock_dot(ket_w, ket_v):
    if np.array_equal(ket_w, ket_v):
        return 1
    else:
        return 0


@njit
def get_density_matrix_comp(
    M, Fock_Basis, evecs, pbc, k=0, statistic="b", component_no=0, component_count=1
):
    values = List()

    pos_i_list = List()

    pos_j_list = List()

    range_end = M

    ket_psi = evecs[:, k]

    ket_len = M

    for i in range(0, range_end):
        for j in range(0, range_end):
            for num_w, ket_w in enumerate(Fock_Basis):
                for num_v, ket_v in enumerate(Fock_Basis):
                    c_ij = 0
                    for (anihillated, created) in [(j, i)]:
                        hash_val = 0
                        factor_1, ket_v_tilde = a_operator(
                            anihillated,
                            ket_v,
                            "a",
                            pbc=pbc,
                            statistic=statistic,
                            component_number=component_no,
                            ket_size=ket_len,
                            component_count=component_count,
                        )
                        factor_2, ket_v_tilde_tilde = a_operator(
                            created,
                            ket_v_tilde,
                            "c",
                            pbc=pbc,
                            statistic=statistic,
                            component_number=component_no,
                            ket_size=ket_len,
                            component_count=component_count,
                        )

                        hash_val = (
                            factor_1
                            * factor_2
                            * fock_dot(ket_w, ket_v_tilde_tilde)
                            * np.conj(ket_psi[num_w])
                            * ket_psi[num_v]
                        )
                        c_ij += hash_val

                    if c_ij == 0:
                        pass
                    else:
                        values.append(float(c_ij))

                        pos_i_list.append(i)

                        pos_j_list.append(j)

    return values, pos_i_list, pos_j_list


def get_density_matrix(
    N,
    M,
    Fock_Basis,
    evecs,
    pbc,
    V,
    J,
    k=0,
    statistic=["b"],
    save_sparse=False,
    save_plot=False,
    plot_show=False,
    annot_plot=False,
    component_count=1,
    component_no=None,
):
    # first for the ground state, k = 0
    # First tab - values, second and third - coordinates
    if component_no is None:
        tmp = sc.sparse.coo_matrix(evecs[:, k])
        rho_k = sc.sparse.coo_matrix(
            sc.sparse.coo_matrix.conjugate(tmp.transpose()) * tmp
        )
        if save_sparse:
            p = (
                pathlib.Path.cwd()
                .joinpath("{}_component".format(component_count))
                .joinpath("Density_Matrices")
                .joinpath("N={}_L={}".format(N, M))
            )
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
            sc.sparse.save_npz(
                "{}_component/Density_Matrices/N={}_L={}/{}_DM_N={}_L={}_V_J=".format(
                    component_count,
                    N,
                    M,
                    "".join(
                        [(lambda el: "B" if el == "b" else "F")(el) for el in statistic]
                    )
                    + "H",
                    N,
                    M,
                )
                + "_".join(
                    [
                        (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                        for el in np.abs(V / J)
                    ]
                )
                + "_k={}".format(k)
                + "_comp_{}_{}".format(component_no, "OBC" if pbc == 0 else "PBC"),
                rho_k,
            )
        if save_plot:
            p_plot = (
                pathlib.Path.cwd()
                .joinpath("{}_component".format(component_count))
                .joinpath("Plots")
                .joinpath("Density_Matrices")
                .joinpath("N={}_L={}".format(N, M))
            )
            if not p_plot.exists():
                p_plot.mkdir(parents=True, exist_ok=True)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            ax.set_title(
                "Stat = "
                + ", ".join([(lambda el: "B" if el == "b" else "F")(el) for el in stat])
                + "; N = "
                + ", ".join([(lambda el: "{}".format(el))(el) for el in N])
                + "; L = {}; V/J = ".format(M)
                + ", ".join(
                    [
                        (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                        for el in np.abs(V / J)
                    ]
                )
            )
            sns.heatmap(
                np.matrix(rho_k.toarray()),
                square=True,
                cmap="coolwarm",
                annot=annot_plot,
                ax=ax,
            )
            fig.savefig(
                "{}_component/Plots/Density_Matrices/N={}_L={}/{}_Hamiltonian_N={}_L={}_V_J=".format(
                    component_count,
                    N,
                    M,
                    "".join(
                        [(lambda el: "B" if el == "b" else "F")(el) for el in statistic]
                    )
                    + "H",
                    N,
                    M,
                )
                + "_".join(
                    [
                        (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                        for el in np.abs(V / J)
                    ]
                )
                + "_comp_{}_k={}_{}.pdf".format(
                    component_no, k, "OBC" if pbc == 0 else "PBC"
                )
            )
            if plot_show:
                plt.show(fig)
            plt.close()
    else:
        rho_k = [List(), List(), List()]
        rho_k = get_density_matrix_comp(
            M,
            Fock_Basis,
            evecs,
            pbc,
            statistic=statistic[component_no],
            component_count=component_count,
            component_no=component_no,
            k=k,
        )
        rho_k = sc.sparse.coo_matrix((rho_k[0], (rho_k[1], rho_k[2])), shape=(M, M))
        if save_sparse:
            p = (
                pathlib.Path.cwd()
                .joinpath("{}_component".format(component_count))
                .joinpath("Density_Matrices")
                .joinpath("N={}_L={}".format(N, M))
            )
            if not p.exists():
                p.mkdir(parents=True, exist_ok=True)
            sc.sparse.save_npz(
                "{}_component/Density_Matrices/N={}_L={}/{}_DM_N={}_L={}_V_J=".format(
                    component_count,
                    N,
                    M,
                    "".join(
                        [(lambda el: "B" if el == "b" else "F")(el) for el in statistic]
                    )
                    + "H",
                    N,
                    M,
                )
                + "_".join(
                    [
                        (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                        for el in np.abs(V / J)
                    ]
                )
                + "_k={}".format(k)
                + "_comp_{}_{}".format(component_no, "OBC" if pbc == 0 else "PBC"),
                rho_k,
            )
        if save_plot:
            p_plot = (
                pathlib.Path.cwd()
                .joinpath("{}_component".format(component_count))
                .joinpath("Plots")
                .joinpath("Density_Matrices")
                .joinpath("N={}_L={}".format(N, M))
            )
            if not p_plot.exists():
                p_plot.mkdir(parents=True, exist_ok=True)
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111)
            ax.set_title(
                "Stat = "
                + ", ".join([(lambda el: "B" if el == "b" else "F")(el) for el in stat])
                + "; N = "
                + ", ".join([(lambda el: "{}".format(el))(el) for el in N])
                + "; L = {}; V/J = ".format(M)
                + ", ".join(
                    [
                        (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                        for el in np.abs(V / J)
                    ]
                )
            )
            sns.heatmap(
                np.matrix(rho_k.toarray()),
                square=True,
                cmap="coolwarm",
                annot=annot_plot,
                ax=ax,
            )
            fig.savefig(
                "{}_component/Plots/Density_Matrices/N={}_L={}/{}_Hamiltonian_N={}_L={}_V_J=".format(
                    component_count,
                    N,
                    M,
                    "".join(
                        [(lambda el: "B" if el == "b" else "F")(el) for el in statistic]
                    )
                    + "H",
                    N,
                    M,
                )
                + "_".join(
                    [
                        (lambda el: re.sub("\\.", ",", "{:.2f}".format(el)))(el)
                        for el in np.abs(V / J)
                    ]
                )
                + "_comp_{}_k={}_{}.pdf".format(
                    component_no, k, "OBC" if pbc == 0 else "PBC"
                )
            )
            if plot_show:
                plt.show(fig)
            plt.close()
    return rho_k


@njit
def _get_n_i_squared(i, Fock_Basis, evecs, pbc=0, k=0, component_count=1):
    values = List()

    pos_i_list = List()

    pos_j_list = List()

    ket_psi = evecs[:, k]

    for num_w, ket_w in enumerate(Fock_Basis):
        for num_v, ket_v in enumerate(Fock_Basis):
            # if pbc == 0:
            #     i_next = i+1
            # else:
            #     i_next = np.mod(i+1, ket.shape[0])
            c_ii = 0
            for (anihillated, created) in [(i, i)]:
                hash_val = 0
                factor_1, ket_v_tilde = a_operator(
                    anihillated, ket_v, "a", pbc=pbc, component_count=component_count
                )
                factor_2, ket_v_2_tilde = a_operator(
                    created, ket_v_tilde, "c", pbc=pbc, component_count=component_count
                )
                factor_3, ket_v_3_tilde = a_operator(
                    anihillated,
                    ket_v_2_tilde,
                    "a",
                    pbc=pbc,
                    component_count=component_count,
                )
                factor_4, ket_v_4_tilde = a_operator(
                    created,
                    ket_v_3_tilde,
                    "c",
                    pbc=pbc,
                    component_count=component_count,
                )

                hash_val = (
                    factor_1
                    * factor_2
                    * factor_3
                    * factor_4
                    * fock_dot(ket_w, ket_v_4_tilde)
                    * np.conj(ket_psi[num_w])
                    * ket_psi[num_v]
                )
                c_ii += hash_val

            if c_ii == 0:
                pass
            else:
                values.append(float(c_ii))

                pos_i_list.append(i)

                pos_j_list.append(i)

    return sum(values)


@njit
def _get_n_i_variance(i, ni_in, Fock_Basis, evecs, pbc=0, k=0):
    ni_sq = _get_n_i_squared(i, Fock_Basis, evecs, pbc, k)
    sq_ni = np.power(ni_in, 2)
    return np.sqrt(ni_sq - sq_ni)


def _get_total_variance(N, rho_k, Fock_Basis, evecs, pbc=0, k=0):
    i_tab = np.arange(0, N)
    var_tot = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        future_to_n_i_variations = {
            executor.submit(
                _get_n_i_variance, i, rho_k.diagonal()[i], Fock_Basis, evecs, pbc, k
            ): i
            for i in i_tab
        }
        for future in concurrent.futures.as_completed(future_to_n_i_variations):
            # print("Completed i = {}, result = {:.5f}".format(future_to_n_i_variations[future], future.result()))
            var_tot += future.result()
        var_tot /= N
    return var_tot


def _get_data_for_plots(M, N, J, V, pbc=0, k=0, basis_apriori=None, plot_type="sparse"):
    if basis_apriori is None:
        Fock_Basis = get_basis(M, N)
    else:
        Fock_Basis = basis_apriori
    if plot_type == "sparse":
        sparse_type = True
        accurate_type = False
        save_parameter = True
    elif plot_type == "off":
        sparse_type = False
        accurate_type = False
        save_parameter = False
    else:
        sparse_type = False
        accurate_type = True
        save_parameter = True
    H = get_Hamiltonian(
        N=N,
        M=M,
        J=J,
        V=V,
        basis_apriori=Fock_Basis,
        pbc=1,
        plot_accurate=accurate_type,
        annot_accurate=False,
        plot_sparsity_pattern=sparse_type,
        save_plot=save_parameter,
        return_basis=False,
        save_sparse=True,
        plot_show=False,
    )
    evals, evecs = sc.sparse.linalg.eigsh(H, 2, which="SA", return_eigenvectors=True)
    delta_E = evals[1] - evals[0]
    rho_k = get_density_matrix(
        N, M, Fock_Basis, evecs, pbc, V, J, save_sparse=True, save_plot=save_parameter
    )
    lambda_1 = sc.sparse.linalg.eigsh(rho_k, 1, which="LA", return_eigenvectors=False)[
        0
    ]
    f_c = lambda_1 / N
    rho_csr = sc.sparse.csr_matrix(rho_k)
    correltation = rho_csr[0, (M // 2) - 1]
    var_tot = _get_total_variance(N, rho_k, Fock_Basis, evecs, pbc, k)
    return {
        "delta_E": delta_E,
        "f_c": f_c,
        "total_variance": var_tot,
        "correlation": correltation,
    }


def _generate_data(N, M, J, V_range=None, pbc=1, plot_type="sparse"):
    if V_range is None:
        V_tab = np.arange(0, 20, 1)
    else:
        V_tab = V_range
    D = calc_Dim(M, N)
    Fock_Basis = get_basis(M, N)
    plot_data = {
        "V/J": [],
        "Delta_E": [],
        "f_c": [],
        "total_variance": [],
        "correlation": [],
    }
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
        future_to_plot_parameters = {
            executor.submit(
                _get_data_for_plots,
                M,
                N,
                J,
                V,
                pbc,
                basis_apriori=Fock_Basis,
                plot_type=plot_type,
            ): V
            for V in V_tab
        }
        for future in concurrent.futures.as_completed(future_to_plot_parameters):
            # result[future_to_plot_parameters[future]] = future.result()
            plot_data["V/J"].append(future_to_plot_parameters[future])
            plot_data["Delta_E"].append(future.result()["delta_E"])
            plot_data["f_c"].append(future.result()["f_c"])
            plot_data["total_variance"].append(future.result()["total_variance"])
            plot_data["correlation"].append(future.result()["correlation"])
    return plot_data


if __name__ == "__main__":
    # N - Number of particles in a system
    N = 2
    # M - Number of sites to fill
    M = 3
    # J - Hopping scaling factor
    J = -1
    # V - On-site potential scaling factor
    V = 1
    # Statistic (bose/fermi)
    stat = "f"
    # Comp no
    comp_no = 2
    # Periodic bounary conditions
    pbc = 0

    # D - Dimension of the final Hamiltonian matrix
    D = calc_Dim(M, N, statistic=stat)
    A = get_tensor_basis(M, N, statistic=stat, n_components=comp_no)
    # Getting the basis vectors hashed
    tab_T = np.array([tag_func(v) for v in A])
    # Preserving the original order of basis vectors
    ind = np.argsort(tab_T)
    # Sorting the new array for more efficient searching (by bisection)
    t_sorted = tab_T.copy()
    t_sorted.sort()
    t_dict = Dict()
    for key, val in zip(tab_T, np.arange(0, A.shape[0])):
        t_dict[key] = val
