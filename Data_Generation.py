#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 23:42:33 2022

File for generating data for the CNN training/validation/testing

@author: k4cp3rskiii
"""

# =============================================================================
# File for generation of datasets for further use with CNN
#
# In order to work, it must be in the same folder as 'Hubbard_aux.py' file,
# unless it gets developed in the PIP-installable module by that time.
# =============================================================================


import pathlib
import warnings

import numpy as np
import scipy as sc
import sympy as sp
from numba.typed import List, Dict
from scipy import integrate
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from matplotlib.offsetbox import AnchoredText

warnings.filterwarnings("ignore")
from termcolor import cprint


from Hubbard_aux import (
    calc_dim_tab,
    get_tensor_basis,
    tag_func,
    get_kinetic_H_vw,
)

# %%

if __name__ == "__main__":
    # """

    sets_to_generate=['training', 'validation', 'test']

    # Control parameters

    ww = 1

    for set_label in sets_to_generate:

# =============================================================================
# Split range for the validation and training datasets
# =============================================================================

        # Training dataset values
        if set_label == 'training':
            vv_tab = np.append(np.linspace(0, 0.8, 500), np.linspace(1.2, 2, 500))

        # Validation dataset values
        elif set_label == 'validation':
            vv_tab = np.append(np.linspace(0.001, 0.801, 125), np.linspace(1.201, 2.001, 125))

        # Test dataset values
        elif set_label == 'test':
            vv_tab = np.linspace(0.002, 2.002, 200)

        else:
            raise ValueError("Incorrect dataset designation entered")

# =============================================================================
# Choose between:
#    1. 'training'
#    2. 'validation'
#    3. 'test'
# =============================================================================


        dataset_designation = set_label

        winding_number_tab = []

        observable_tab = []

        w_tab_data = []

        v_tab_data = []

        W = 0.5

        # N - Number of particles in a system
        N = np.array([1])  # If number - the same number for both systems
        #
        # M - Number of sites to fill
        # M = 50
        M = 80
        # J - Hopping scaling factor
        J = 1

        # Number of basis vector components
        component_count = 1

        # Statistic (bose/fermi)
        stat_vec = np.array(['f'])

        n_edge_cells = 4

    # =============================================================================
    #     In this file we will first calculate the winding number for the given v,
    #     using periodic boundary conditions, and then calculate the same system
    #     assuming open boundary conditions, as only in this scenario we can see
    #     the edge states, which we would like to investigate
    # =============================================================================

        cprint("[INFO]", "magenta", end=" ")
        print(f"W = {W} | dataset = {set_label}_set | size = {vv_tab.shape[0]}")

        for vv in tqdm(vv_tab):

        # =========================================================================
        #     Part responsible for calculating the winding number for the system
        # =========================================================================

            # Periodic bounary conditions
            pbc = 1
            # Staggering tab - SSH parameter
            vw_tab = np.array([vv, ww])

            # Disorder Measure - W_1, W_2
            W_disorder_tab = np.array([0.5*W, W])

            p_plot = (
                pathlib.Path.cwd()
                .joinpath("Datasets")
            )
            if not p_plot.exists():
                p_plot.mkdir(parents=True, exist_ok=True)

            stat = stat_vec

            # D - Dimension of the final Hamiltonian matrix
            D = calc_dim_tab(M, N, stat_tab=stat)
            # D = 10
            A = get_tensor_basis(M, N, statistic=stat,
                                 component_count=component_count, verb=0)

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

            H_hop_1 = [List(), List(), List()]
            H_hop_1 = get_kinetic_H_vw(
                A,
                M,
                J,
                t_dict,
                vw=vw_tab,
                disord_W=W_disorder_tab,
                pbc=pbc,
                statistic=stat[0],
                component_count=component_count,
                component_no=0,
            )
            H_hop_1 = sc.sparse.coo_matrix(
                (H_hop_1[0], (H_hop_1[1], H_hop_1[2])), shape=(D, D)
            )

            H_hop_1 = sc.sparse.triu(H_hop_1)

            H_hop_1 = H_hop_1 + H_hop_1.T - \
                sc.sparse.diags(H_hop_1.diagonal(), format='coo')

            H_1 = J*H_hop_1

            H_tot = H_1

            H_dense = H_tot.toarray()

            dvec = np.array([(lambda snum: 1 if snum % 2 == 0 else -1)(snum)
                            for snum in range(M)])

            chiral = sc.sparse.coo_matrix(sc.sparse.diags(dvec))

            ch_evals, ch_evecs = np.linalg.eigh(chiral.toarray())

            sorted_evecs = np.zeros(shape=(D, D))
            sorted_evals = np.zeros(shape=(D,))

            for num, i in enumerate(np.repeat(np.linspace(0,
                                                          D//2,
                                                          D//2,
                                                          endpoint=False,
                                                          dtype=np.int64), 2)):
                rownum = np.where(ch_evecs[:, -(num+1)] == 1)[0][0]
                fact = 0 if num % 2 == 0 else int(D/2)
                new_rownum = fact+i
                new_row = ch_evecs[rownum].copy()
                sorted_evecs[new_rownum] = new_row
                sorted_evals[new_rownum] = ch_evals[rownum].copy()

            ch_evecs = sorted_evecs
            ch_evals = sorted_evals

            H_new_basis = ch_evecs@H_dense@(np.linalg.inv(ch_evecs))

            u = H_new_basis[D//2:, :D//2]

            v, w, k = sp.symbols('v w k', real=True)

            u = sp.Matrix(u)

            u[-1, 0] = u[-1, 0]*sp.exp(sp.I*k)

            dkh = sp.simplify(sp.Derivative(u, k).doit())

            hinv = sp.Inverse(u).doit()

            h1_fin = hinv@dkh

            expr1 = sp.Trace(h1_fin).doit()/(2*sp.pi*sp.I)

            integrand1 = sp.lambdify(k, expr1.doit(), 'scipy')

            r1, r1err = integrate.quad(integrand1, -np.pi, np.pi-np.finfo(float).eps)

            winding_number_tab.append(r1)

    # =========================================================================
    #     Part responsible for calculating the eigenvectors observables to feed
    #     the CNN with
    # =========================================================================

            # Periodic bounary conditions
            pbc = 0

            H_hop_obc = [List(), List(), List()]
            H_hop_obc = get_kinetic_H_vw(
                A,
                M,
                J,
                t_dict,
                vw=vw_tab,
                disord_W=W_disorder_tab,
                pbc=pbc,
                statistic=stat[0],
                component_count=component_count,
                component_no=0,
            )
            H_hop_obc = sc.sparse.coo_matrix(
                (H_hop_obc[0], (H_hop_obc[1], H_hop_obc[2])), shape=(D, D)
            )

            H_hop_obc = sc.sparse.triu(H_hop_obc)

            H_hop_obc = H_hop_obc + H_hop_obc.T - \
                sc.sparse.diags(H_hop_obc.diagonal(), format='coo')

            H_obc = J*H_hop_obc

            H_tot_obc = H_obc

            H_dense_obc = H_tot_obc.toarray()

            evals_obc, evecs_obc = np.linalg.eigh(H_dense_obc)

            observable = evecs_obc**2

            observable_tab.append(observable)

            w_tab_data.append(W)

            v_tab_data.append(vv)


        # fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        # fig.suptitle("Numerical integration", fontsize=20)
        # plt.tight_layout()
        # ax.plot(vv_tab, winding_number_tab, lw=0, marker=6,
        #         label=
        #         r"$\frac{1}{2 \pi i} \int_{-\pi}^{\pi} Tr[h^{-1} \partial_k h]$ dk")
        # ax.set_ylabel("Winding Number $\mathcal{W}$", fontsize=18)
        # ax.set_xlabel("$\mathcal{v}$", fontsize=18)
        # ax.set_yticks([0, 1])
        # ax.axvline(x=1.0, ls="dotted")
        # at = AnchoredText(
        #     f"w={ww}, L={M}, W={W}", prop=dict(size=20), frameon=True,
        #     loc='lower left')
        # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        # ax.add_artist(at)
        # plt.legend(loc='upper right', fontsize=25)

        res_dict = {"W": w_tab_data, "labels": np.round(
            np.abs(winding_number_tab)).astype(dtype=np.int64), "data": observable_tab, "v": v_tab_data}

        with open(f"Datasets/{dataset_designation}_set.pickle", "wb") as f:
            pickle.dump(res_dict, f)



"""

# =============================================================================
#   An alteration of program, used to generate just one of the sets.
#   Uncomment the block below to get to it
# =============================================================================

"""
"""
# Control parameters

    ww = 1



# =============================================================================
# Split range for the validation and training datasets
# =============================================================================

    # Training dataset values
    # vv_tab = np.append(np.linspace(0, 0.8, 500), np.linspace(1.2, 2, 500))

    # Validation dataset values
    # vv_tab = np.append(np.linspace(0.001, 0.801, 125), np.linspace(1.201, 2.001, 125))

    # Test dataset values
    vv_tab = np.linspace(0.002, 2.002, 200)

# =============================================================================
# Choose between:
#    1. 'training'
#    2. 'validation'
#    3. 'test'
# =============================================================================


    dataset_designation = 'test'

    winding_number_tab = []
    observable_tab = []

    w_tab_data = []

    v_tab_data = []

    W = 1.5


    # N - Number of particles in a system
    N = np.array([1])  # If number - the same number for both systems
    #
    # M - Number of sites to fill
    M = 50
    # M = 80
    # J - Hopping scaling factor
    J = 1

    # Number of basis vector components
    component_count = 1

    # Statistic (bose/fermi)
    stat_vec = np.array(['f'])

    n_edge_cells = 4

# =============================================================================
#     In this file we will first calculate the winding number for the given v,
#     using periodic boundary conditions, and then calculate the same system
#     assuming open boundary conditions, as only in this scenario we can see
#     the edge states, which we would like to investigate
# =============================================================================


    for vv in tqdm(vv_tab):

    # =========================================================================
    #     Part responsible for calculating the winding number for the system
    # =========================================================================

        # Periodic bounary conditions
        pbc = 1
        # Staggering tab - SSH parameter
        vw_tab = np.array([vv, ww])

        # Disorder Measure - W_1, W_2
        W_disorder_tab = np.array([0.5*W, W])

        p_plot = (
            pathlib.Path.cwd()
            .joinpath("Datasets")
        )
        if not p_plot.exists():
            p_plot.mkdir(parents=True, exist_ok=True)

        stat = stat_vec

        # D - Dimension of the final Hamiltonian matrix
        D = calc_dim_tab(M, N, stat_tab=stat)
        # D = 10
        A = get_tensor_basis(M, N, statistic=stat,
                             component_count=component_count, verb=0)

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

        H_hop_1 = [List(), List(), List()]
        H_hop_1 = get_kinetic_H_vw(
            A,
            M,
            J,
            t_dict,
            vw=vw_tab,
            disord_W=W_disorder_tab,
            pbc=pbc,
            statistic=stat[0],
            component_count=component_count,
            component_no=0,
        )
        H_hop_1 = sc.sparse.coo_matrix(
            (H_hop_1[0], (H_hop_1[1], H_hop_1[2])), shape=(D, D)
        )

        H_hop_1 = sc.sparse.triu(H_hop_1)

        H_hop_1 = H_hop_1 + H_hop_1.T - \
            sc.sparse.diags(H_hop_1.diagonal(), format='coo')

        H_1 = J*H_hop_1

        H_tot = H_1

        H_dense = H_tot.toarray()

        dvec = np.array([(lambda snum: 1 if snum % 2 == 0 else -1)(snum)
                        for snum in range(M)])

        chiral = sc.sparse.coo_matrix(sc.sparse.diags(dvec))

        ch_evals, ch_evecs = np.linalg.eigh(chiral.toarray())

        sorted_evecs = np.zeros(shape=(D, D))
        sorted_evals = np.zeros(shape=(D,))

        for num, i in enumerate(np.repeat(np.linspace(0,
                                                      D//2,
                                                      D//2,
                                                      endpoint=False,
                                                      dtype=np.int64), 2)):
            rownum = np.where(ch_evecs[:, -(num+1)] == 1)[0][0]
            fact = 0 if num % 2 == 0 else int(D/2)
            new_rownum = fact+i
            new_row = ch_evecs[rownum].copy()
            sorted_evecs[new_rownum] = new_row
            sorted_evals[new_rownum] = ch_evals[rownum].copy()

        ch_evecs = sorted_evecs
        ch_evals = sorted_evals

        H_new_basis = ch_evecs@H_dense@(np.linalg.inv(ch_evecs))

        u = H_new_basis[D//2:, :D//2]

        v, w, k = sp.symbols('v w k', real=True)

        u = sp.Matrix(u)

        u[-1, 0] = u[-1, 0]*sp.exp(sp.I*k)

        dkh = sp.simplify(sp.Derivative(u, k).doit())

        hinv = sp.Inverse(u).doit()

        h1_fin = hinv@dkh

        expr1 = sp.Trace(h1_fin).doit()/(2*sp.pi*sp.I)

        integrand1 = sp.lambdify(k, expr1.doit(), 'scipy')

        r1, r1err = integrate.quad(integrand1, -np.pi, np.pi-np.finfo(float).eps)

        winding_number_tab.append(r1)

    # =========================================================================
    #     Part responsible for calculating the eigenvectors observables to feed
    #     the CNN with
    # =========================================================================

        # Periodic bounary conditions
        pbc = 0

        H_hop_obc = [List(), List(), List()]
        H_hop_obc = get_kinetic_H_vw(
            A,
            M,
            J,
            t_dict,
            vw=vw_tab,
            disord_W=W_disorder_tab,
            pbc=pbc,
            statistic=stat[0],
            component_count=component_count,
            component_no=0,
        )
        H_hop_obc = sc.sparse.coo_matrix(
            (H_hop_obc[0], (H_hop_obc[1], H_hop_obc[2])), shape=(D, D)
        )

        H_hop_obc = sc.sparse.triu(H_hop_obc)

        H_hop_obc = H_hop_obc + H_hop_obc.T - \
            sc.sparse.diags(H_hop_obc.diagonal(), format='coo')

        H_obc = J*H_hop_obc

        H_tot_obc = H_obc

        H_dense_obc = H_tot_obc.toarray()

        evals_obc, evecs_obc = np.linalg.eigh(H_dense_obc)

        observable = evecs_obc**2

        observable_tab.append(observable)

        w_tab_data.append(W)

        v_tab_data.append(vv)


    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    fig.suptitle("Numerical integration", fontsize=20)
    plt.tight_layout()
    ax.plot(vv_tab, winding_number_tab, lw=0, marker=6,
            label=
            r"$\frac{1}{2 \pi i} \int_{-\pi}^{\pi} Tr[h^{-1} \partial_k h]$ dk")
    ax.set_ylabel("Winding Number $\mathcal{W}$", fontsize=18)
    ax.set_xlabel("$\mathcal{v}$", fontsize=18)
    ax.set_yticks([0, 1])
    ax.axvline(x=1.0, ls="dotted")
    at = AnchoredText(
        f"w={ww}, L={M}, W={W}", prop=dict(size=20), frameon=True,
        loc='lower left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    plt.legend(loc='upper right', fontsize=25)

    res_dict = {"W": w_tab_data, "labels": np.round(
        np.abs(winding_number_tab)).astype(dtype=np.int64), "data": observable_tab, "v": v_tab_data}

    with open(f"Datasets/{dataset_designation}_set.pickle", "wb") as f:
        pickle.dump(res_dict, f)

"""
"""
# =============================================================================
#   An alteration of program, used to generate few realisations of
#   just one of the sets.
#   Uncomment the block below to get to it
# =============================================================================

if __name__ == "__main__":
    # W = 1

    w_pre = 0
    n_realizations = 10
    counter_start = 40

    for W in [
        # w_pre + 0.0,
        # w_pre + 0.1,
        # w_pre + 0.2,
        w_pre + 0.3,
        w_pre + 0.4,
        w_pre + 0.5,
        w_pre + 0.6,
        w_pre + 0.7,
        w_pre + 0.8,
        w_pre + 0.9,
    ]:
        cprint("[INFO]", "magenta", end=" ")
        print(f"W = {W}")

        for real_num in tqdm(range(n_realizations), colour="green", unit="Realisation"):
            # Control parameters

            ww = 1

            # =============================================================================
            # Split range for the validation and training datasets
            # =============================================================================

            # Training dataset values
            # vv_tab = np.append(np.linspace(0, 0.8, 500), np.linspace(1.2, 2, 500))

            # Validation dataset values
            # vv_tab = np.append(np.linspace(0.001, 0.801, 125), np.linspace(1.201, 2.001, 125))

            # Test dataset values
            vv_tab = np.linspace(0.002, 2.002, 500)

            # =============================================================================
            # Choose between:
            #    1. 'training'
            #    2. 'validation'
            #    3. 'test'
            # =============================================================================

            dataset_designation = f"test_{counter_start+real_num}"

            winding_number_tab = []
            observable_tab = []

            w_tab_data = []

            v_tab_data = []

            # N - Number of particles in a system
            N = np.array([1])  # If number - the same number for both systems
            #
            # M - Number of sites to fill
            M = 50
            # M = 80
            # J - Hopping scaling factor
            J = 1

            # Number of basis vector components
            component_count = 1

            # Statistic (bose/fermi)
            stat_vec = np.array(["f"])

            n_edge_cells = 4

            # =============================================================================
            #     In this file we will first calculate the winding number for the given v,
            #     using periodic boundary conditions, and then calculate the same system
            #     assuming open boundary conditions, as only in this scenario we can see
            #     the edge states, which we would like to investigate
            # =============================================================================

            for vv in tqdm(vv_tab, colour="cyan", unit="H"):
                # =========================================================================
                #     Part responsible for calculating the winding number for the system
                # =========================================================================

                # Periodic bounary conditions
                pbc = 1
                # Staggering tab - SSH parameter
                vw_tab = np.array([vv, ww])

                # Disorder Measure - W_1, W_2
                W_disorder_tab = np.array([0.5 * W, W])

                p_plot = pathlib.Path.cwd().joinpath("Datasets")
                if not p_plot.exists():
                    p_plot.mkdir(parents=True, exist_ok=True)

                stat = stat_vec

                # D - Dimension of the final Hamiltonian matrix
                D = calc_dim_tab(M, N, stat_tab=stat)
                # D = 10
                A = get_tensor_basis(
                    M, N, statistic=stat, component_count=component_count, verb=0
                )

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

                H_hop_1 = [List(), List(), List()]
                H_hop_1 = get_kinetic_H_vw(
                    A,
                    M,
                    J,
                    t_dict,
                    vw=vw_tab,
                    disord_W=W_disorder_tab,
                    pbc=pbc,
                    statistic=stat[0],
                    component_count=component_count,
                    component_no=0,
                )
                H_hop_1 = sc.sparse.coo_matrix(
                    (H_hop_1[0], (H_hop_1[1], H_hop_1[2])), shape=(D, D)
                )

                H_hop_1 = sc.sparse.triu(H_hop_1)

                H_hop_1 = (
                    H_hop_1
                    + H_hop_1.T
                    - sc.sparse.diags(H_hop_1.diagonal(), format="coo")
                )

                H_1 = J * H_hop_1

                H_tot = H_1

                H_dense = H_tot.toarray()

                dvec = np.array(
                    [
                        (lambda snum: 1 if snum % 2 == 0 else -1)(snum)
                        for snum in range(M)
                    ]
                )

                chiral = sc.sparse.coo_matrix(sc.sparse.diags(dvec))

                ch_evals, ch_evecs = np.linalg.eigh(chiral.toarray())

                sorted_evecs = np.zeros(shape=(D, D))
                sorted_evals = np.zeros(shape=(D,))

                for num, i in enumerate(
                    np.repeat(
                        np.linspace(0, D // 2, D // 2, endpoint=False, dtype=np.int64),
                        2,
                    )
                ):
                    rownum = np.where(ch_evecs[:, -(num + 1)] == 1)[0][0]
                    fact = 0 if num % 2 == 0 else int(D / 2)
                    new_rownum = fact + i
                    new_row = ch_evecs[rownum].copy()
                    sorted_evecs[new_rownum] = new_row
                    sorted_evals[new_rownum] = ch_evals[rownum].copy()

                ch_evecs = sorted_evecs
                ch_evals = sorted_evals

                H_new_basis = ch_evecs @ H_dense @ (np.linalg.inv(ch_evecs))

                u = H_new_basis[D // 2 :, : D // 2]

                v, w, k = sp.symbols("v w k", real=True)

                u = sp.Matrix(u)

                u[-1, 0] = u[-1, 0] * sp.exp(sp.I * k)

                dkh = sp.simplify(sp.Derivative(u, k).doit())

                hinv = sp.Inverse(u).doit()

                h1_fin = hinv @ dkh

                expr1 = sp.Trace(h1_fin).doit() / (2 * sp.pi * sp.I)

                integrand1 = sp.lambdify(k, expr1.doit(), "scipy")

                r1, r1err = integrate.quad(
                    integrand1, -np.pi, np.pi - np.finfo(float).eps
                )

                winding_number_tab.append(r1)

                del r1, r1err, hinv, h1_fin, expr1, integrand1, u, H_new_basis, H_1, H_dense, H_hop_1

                # =========================================================================
                #     Part responsible for calculating the eigenvectors observables to feed
                #     the CNN with
                # =========================================================================

                # Periodic bounary conditions
                pbc = 0

                H_hop_obc = [List(), List(), List()]
                H_hop_obc = get_kinetic_H_vw(
                    A,
                    M,
                    J,
                    t_dict,
                    vw=vw_tab,
                    disord_W=W_disorder_tab,
                    pbc=pbc,
                    statistic=stat[0],
                    component_count=component_count,
                    component_no=0,
                )
                H_hop_obc = sc.sparse.coo_matrix(
                    (H_hop_obc[0], (H_hop_obc[1], H_hop_obc[2])), shape=(D, D)
                )

                H_hop_obc = sc.sparse.triu(H_hop_obc)

                H_hop_obc = (
                    H_hop_obc
                    + H_hop_obc.T
                    - sc.sparse.diags(H_hop_obc.diagonal(), format="coo")
                )

                H_obc = J * H_hop_obc

                H_tot_obc = H_obc

                H_dense_obc = H_tot_obc.toarray()

                evals_obc, evecs_obc = np.linalg.eigh(H_dense_obc)

                observable = evecs_obc**2

                observable_tab.append(observable)

                w_tab_data.append(W)

                v_tab_data.append(vv)

            res_dict = {
                "W": w_tab_data,
                "labels": np.round(np.abs(winding_number_tab)).astype(dtype=np.int64),
                "data": observable_tab,
                "v": v_tab_data,
            }

            with open(f"Datasets/W={W}/{dataset_designation}_set.pickle", "wb") as f:
                pickle.dump(res_dict, f)

            del res_dict, winding_number_tab, w_tab_data, observable_tab, v_tab_data, H_dense_obc, observable, H_obc, H_hop_obc, H_tot_obc, evals_obc, evecs_obc, t_dict

"""
