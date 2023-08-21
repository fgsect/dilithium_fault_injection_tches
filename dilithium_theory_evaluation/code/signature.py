#!/usr/bin/python3

from collections import namedtuple
import logging
from parameters import Parameters
import numpy as np
from numpy import typing as npt
from scipy.linalg import toeplitz, null_space

Signature = namedtuple('Signature', ['z', 'c', 'c_matrix'])
DebugSignature = namedtuple('DebugSignature', ['signature', 'y'])
Attack = namedtuple('Attack', ['debug_signatures', 's_1', 'total_faults_per_entry', 'false_positive_rate_per_entry'])


def calculate_c_matrix_np(c: npt.ArrayLike, params: Parameters):
    """
    Assume we have c * s = z, where c, s and z are polynomials over ...
    We can rewrite this equation in the form of As' = b where A is a Matrix and s' and b are the vectors of coefficients of the polynomials s and z respectively.
    This function returns A.
    Adapted from: https://github.com/KatinkaBou/Probabilistic-Bounds-On-Singular-Values-Of-Rotation-Matrices/blob/c92bfa863fc640ca0c39b321dde1696edf84d467/negacyclic_probabilistic_bound.py#L20
    """
    row = np.zeros(c.shape[0], dtype=params.dtype)
    row[0] = c[0]
    row[1:] = -c[-1:0:-1]

    c_matrix = toeplitz(c, row)
    return c_matrix


def gen_y_np(m: int, params: Parameters) -> npt.ArrayLike:
    y_np_flat = np.concatenate((
        np.random.randint(params.y_range.start, params.y_range.stop, m, dtype=params.dtype),  # m not faulted entries
        np.zeros(params.n * params.l - m, dtype=params.dtype)  # nl - m faulted, zeroed, entries
    ))
    np.random.shuffle(y_np_flat)  # all shuffled nicely
    y_np = np.reshape(y_np_flat, (params.l, params.n))  # and put in proper shape
    return y_np


def gen_s_1_np(params: Parameters) -> npt.ArrayLike:
    return np.random.randint(params.s_1_range.start, params.s_1_range.stop, (params.l, params.n), dtype=params.dtype)


def gen_c_np(params: Parameters):
    plus_minus_ones = np.full(params.tau, -1, dtype=params.dtype) ** np.random.randint(0, 2, params.tau, dtype=params.dtype)  # tau many
    zeroes = np.zeros(params.n - params.tau, dtype=params.dtype)  # n - tau many
    c_np = np.concatenate((plus_minus_ones, zeroes))
    np.random.shuffle(c_np)  # shuffled of course

    return c_np


def calculate_z_np(y: npt.ArrayLike, c: npt.ArrayLike, s_1: npt.ArrayLike, params: Parameters):
    if np.shape(c) == (params.n,):
        c_matrix = calculate_c_matrix_np(c, params)
    else:
        assert np.shape(c) == (params.n, params.n)
        c_matrix = c

    z = np.zeros((params.l, params.n), dtype=params.dtype)
    for i in range(params.l):
        np.dot(c_matrix, s_1[i, :], z[i, :])
    z += y

    return z, c_matrix


def gen_sigs_faulty_np(m: int, amount: int, params: Parameters, s_1: npt.ArrayLike = None):
    assert 0 <= m < params.n * params.l
    if s_1 is None:
        s_1 = gen_s_1_np(params)

    total_faults_per_entry = np.zeros((params.l,))
    sigs = []
    for _ in range(amount):
        y = gen_y_np(m, params)
        c = gen_c_np(params)
        z, c_matrix = calculate_z_np(y, c, s_1, params)

        sigs.append(DebugSignature(Signature(z, c, c_matrix), y))
        total_faults_per_entry += np.count_nonzero(y == 0, axis=1)

    for i, num_faults in enumerate(total_faults_per_entry):
        logging.info(f'Entry {i} of s_1 does have {num_faults} faults.')
        if num_faults < params.n:
            logging.warning(f'Entry {i} of s_1 does not have enough faults! Attack will most likely not work. Only {num_faults} faults are present.')

    return sigs, s_1, total_faults_per_entry


def gen_sigs_until_success(m: int, threshold: int, params: Parameters, notion_of_success: int = None):
    assert 0 <= m < params.n * params.l

    if notion_of_success is None:
        notion_of_success = params.n
    assert notion_of_success >= params.n

    s_1 = gen_s_1_np(params)

    num_total_after_filter = np.zeros(params.l)
    num_total_before_filter = np.zeros(params.l)
    big_c_matrices = [np.zeros((0, params.n)) for _ in range(params.l)]
    big_z_vectors = [np.zeros(0) for _ in range(params.l)]
    sigs = []
    while True:
        y = gen_y_np(m, params)
        c = gen_c_np(params)
        z, c_matrix = calculate_z_np(y, c, s_1, params)

        sigs.append(DebugSignature(Signature(z, c, c_matrix), y))

        for i in range(params.l):
            is_zero_mask = (y[i] == 0) & (np.absolute(z[i]) <= threshold)
            big_c_matrices[i] = np.vstack((big_c_matrices[i], c_matrix[is_zero_mask]))
            big_z_vectors[i] = np.concatenate((big_z_vectors[i], z[i][is_zero_mask]))
            num_total_after_filter[i] += np.shape(y[i][np.absolute(z[i]) <= threshold])[0]
            num_total_before_filter[i] += np.shape(y)[0]

        # do we have at least 256 equations per polynomial? if not we continue
        if not all([big_c_matrices[i].shape[0] >= notion_of_success for i in range(params.l)]):
            continue

        # if we have enough equations it might be the case that they are not linear independent
        # I do not know how likely that is but just to be sure
        do_continue = False
        for i in range(params.l):
            x, _, rank, _ = np.linalg.lstsq(big_c_matrices[i], big_z_vectors[i], rcond=None)
            is_hom = np.all(big_z_vectors[i] == 0)
            if not is_hom:
                if rank < params.n:
                    logging.info(f'We have the case of linear dependent equations for entry number {i}: rank = {rank}, num_equations={big_c_matrices[i].shape[0]}')
                    do_continue = True
                break
            else:
                # okay, we have hom system
                if rank < params.n - 1:
                    logging.info(
                        f'We have the case of homo system and rank is not n - 1 for entry number {i}: rank = {rank}, num_equations={big_c_matrices[i].shape[0]}')
                    do_continue = True
                    break
            if not is_hom:  # only in this case we have unique solution
                s_1_i = np.around(x).astype(params.dtype)
                if not np.all(s_1[i] == s_1_i):
                    logging.error(f'We have full rank but the recovery is wrong entry {i}! This should not happen when location of zero is known. (rank={rank})')
            else:
                logging.debug(f'Recovering s_1_{i} from hom system')
                null = null_space(big_c_matrices[i])
                assert np.shape(null) == (256, 1)
                null_vec = np.transpose(null[:, 0])
                assert np.shape(null_vec) == (256,)

                # we try to get a non-null entry, not guaranteed, but _very_ likely
                non_null_entry = max(np.abs(np.min(null_vec)), np.max(null_vec))
                hom_is_alright = False
                for coefficient_guess in params.s_1_range:
                    scale = coefficient_guess / non_null_entry
                    solution_candidate = np.around(scale * null_vec).astype(params.dtype)
                    if np.all(solution_candidate == s_1[i]):
                        hom_is_alright = True
                        logging.debug('Hom system alright!')
                        break
                assert hom_is_alright

        if do_continue:
            continue
        break

    num_zeros_after_filter = np.array([big_c_matrices[i].shape[0] for i in range(params.l)], dtype=params.dtype)
    false_positive_rate = (num_total_after_filter - num_zeros_after_filter) / num_total_after_filter
    # print(num_zeros_after_filter)
    # print(num_total_after_filter)
    # print(num_total_before_filter)
    return Attack(sigs, s_1, num_zeros_after_filter, false_positive_rate)
