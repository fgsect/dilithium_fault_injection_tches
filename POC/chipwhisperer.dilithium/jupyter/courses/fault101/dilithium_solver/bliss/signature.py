#!/usr/bin/python3

from collections import namedtuple
import logging

import scipy.stats

from parameters import Parameters
import numpy as np
from numpy import typing as npt
from scipy.linalg import toeplitz

Signature = namedtuple('Signature', ['z_1', 'c', 'c_matrix'])
DebugSignature = namedtuple('DebugSignature', ['signature', 'y_1', 'sign'])
Attack = namedtuple('Attack', ['debug_signatures', 's_1', 'total_faults'])


def calculate_z_1_matrix(z_1s: list[npt.ArrayLike], params: Parameters) -> npt.ArrayLike:
    matrix = np.zeros((sum([np.shape(z_1)[0] for z_1 in z_1s]), len(z_1s)), dtype=params.dtype)

    for i in range(len(z_1s)):
        next_column = np.concatenate((
            np.zeros(sum([np.shape(z_1)[0] for z_1 in z_1s[:i]]), dtype=params.dtype),
            2 * z_1s[i],
            np.zeros(sum([np.shape(z_1)[0] for z_1 in z_1s[i + 1:]]), dtype=params.dtype),
        ))
        matrix[:, i] = next_column

    return matrix


def calculate_c_matrix_np(c: npt.ArrayLike, params: Parameters) -> npt.ArrayLike:
    """
    Assume we have c * s = z, where c, s and z are polynomials over ...
    We can rewrite this equation in the form of As' = b where A is a Matrix and s' and b are the vectors of coefficients of the polynomials s and z respectively.
    This function returns A.
    Adapted from: https://github.com/KatinkaBou/Probabilistic-Bounds-On-Singular-Values-Of-Rotation-Matrices/blob/c92bfa863fc640ca0c39b321dde1696edf84d467/negacyclic_probabilistic_bound.py#L20
    """
    row = np.zeros(np.shape(c)[0], dtype=params.dtype)
    row[0] = c[0]
    row[1:] = -c[-1:0:-1]

    c_matrix = toeplitz(c, row)
    return c_matrix


def gen_y_1(m: int, params: Parameters) -> npt.ArrayLike:
    if not hasattr(gen_y_1, "discrete_gaussians"):
        gen_y_1.discrete_gaussians = dict()
    while True:
        try:
            discrete_gaussian = gen_y_1.discrete_gaussians[params]
            break
        except KeyError:
            a = min(params.y_1_range)
            b = max(params.y_1_range)
            x = np.arange(a, b + 1, dtype=params.dtype)
            pmf = np.exp(- x ** 2 / params.sigma ** 2 / 2) / (params.sigma * np.sqrt(2 * np.pi))
            values = (x, pmf)
            gen_y_1.discrete_gaussians[params] = scipy.stats.rv_discrete(a=a, b=b, values=values).freeze()

    non_faulted_coefficients = discrete_gaussian.rvs(size=m)
    faulted_coefficients = np.zeros(params.n - m)
    y_1 = np.concatenate((non_faulted_coefficients, faulted_coefficients))
    np.random.shuffle(y_1)
    return y_1


def gen_s_1(params: Parameters) -> npt.ArrayLike:
    plus_minus_one_coefficients = (-1) ** np.random.randint(0, 1 + 1, params.num_plus_minus_one_coefficients, dtype=params.dtype)
    plus_minus_two_coefficients = 2 * (-1) ** np.random.randint(0, 1 + 1, params.num_plus_minus_two_coefficients, dtype=params.dtype)
    zero_coefficients = np.zeros(params.num_zero_coefficients, dtype=params.dtype)
    s_1 = np.concatenate((plus_minus_one_coefficients, plus_minus_two_coefficients, zero_coefficients))
    np.random.shuffle(s_1)
    return s_1


def gen_c(params: Parameters) -> npt.ArrayLike:
    one_coefficients = np.ones(params.kappa, dtype=params.dtype)
    zero_coefficients = np.zeros(params.n - params.kappa, dtype=params.dtype)
    c = np.concatenate((one_coefficients, zero_coefficients))
    np.random.shuffle(c)
    return c


def calculate_z_1(y_1: npt.ArrayLike, c: npt.ArrayLike, s_1: npt.ArrayLike, params: Parameters):
    c_matrix = calculate_c_matrix_np(c, params)

    c_s_1 = np.dot(c_matrix, s_1)
    sign = (-1) ** np.random.randint(2)
    z = y_1 + sign * c_s_1
    # z = y_1 + c_s_1

    return z, c_matrix, sign


def gen_sigs_until_success(m: int, threshold: int, params: Parameters, notion_of_success: int = None):
    if notion_of_success is None:
        notion_of_success = params.n

    assert 0 <= m < params.n
    assert notion_of_success >= params.n

    s_1 = gen_s_1(params)

    big_c_matrix = np.zeros((0, params.n))
    big_z_vector = np.zeros(0)
    signs = []
    equations_per_signature = []
    debug_signatures = []
    while True:
        y_1 = gen_y_1(m, params)
        c = gen_c(params)
        z_1, c_matrix, sign = calculate_z_1(y_1, c, s_1, params)

        debug_signatures.append(DebugSignature(Signature(z_1, c, c_matrix), y_1, sign))
        signs.append(sign)

        is_zero_mask = (y_1 == 0) & (np.absolute(z_1) <= threshold)
        big_c_matrix = np.vstack((big_c_matrix, c_matrix[is_zero_mask]))
        big_z_vector = np.concatenate((big_z_vector, z_1[is_zero_mask]))
        equations_per_signature.append(np.sum(is_zero_mask))

        # do we have at least 256 equations per polynomial? if not we continue
        if not np.shape(big_c_matrix)[0] >= notion_of_success:
            continue

        # if we have enough equations it might be the case that they are not linear independent
        binary_signs = [0 if s == -1 else s for s in signs]  # 1 -> 1, -1 -> 0
        z_1_correctly_signed = np.dot(
            calculate_z_1_matrix([ds.signature.z_1[ds.y_1 == 0] for ds in debug_signatures], params),
            binary_signs) - np.concatenate([ds.signature.z_1[ds.y_1 == 0] for ds in debug_signatures])
        x, _, rank, _ = np.linalg.lstsq(big_c_matrix, z_1_correctly_signed, rcond=None)
        if rank < params.n:
            logging.info(f'We have the case of linear dependent equations: rank = {rank}, num_equations={big_c_matrix.shape[0]}')
            continue
        s_1_recovered = np.around(x).astype(params.dtype)
        if not np.all(s_1 == s_1_recovered):
            logging.error(f'We have full rank but the recovery is wrong! This should not happen when location of zero is known. (rank={rank})')
            assert np.all(s_1 == s_1_recovered)
        break

    return Attack(debug_signatures, s_1, np.shape(big_c_matrix)[0])
