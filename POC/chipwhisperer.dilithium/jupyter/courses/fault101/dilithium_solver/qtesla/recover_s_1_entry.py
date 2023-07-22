import logging
from typing import List, Tuple, Optional
import numpy as np
import gurobipy as gp
from datetime import datetime
from parameters import Parameters
import numpy.typing as npt
from result import ResultKeys
from signature import DebugSignature
from scipy.stats import hypergeom


def solve_max_rows_simplified(C, z, params: Parameters, timeout: float, oracle) -> Tuple[Optional[npt.ArrayLike], Optional[npt.ArrayLike], dict]:
    GUROBI_STATUS_CODE_TO_STRING = {gp.StatusConstClass.__dict__[k]: k for k in gp.StatusConstClass.__dict__.keys() if
                                    'A' <= k[0] <= 'Z'}

    mdl = gp.Model()

    x = mdl.addMVar(shape=np.shape(z), vtype=gp.GRB.BINARY)
    s = mdl.addMVar(shape=params.n, lb=params.s_1_range.start, ub=params.s_1_range.stop - 1, vtype=gp.GRB.INTEGER)

    mdl.setObjective(x.sum(), gp.GRB.MAXIMIZE)

    mdl.addConstr(z - C @ s <= params.max_diff * (1 - x))
    mdl.addConstr(z - C @ s >= -params.max_diff * (1 - x))

    mdl.params.TimeLimit = timeout
    mdl.params.Threads = params.threads
    logging.info(f'We are using {mdl.params.Threads} threads.')

    mdl._oracle = oracle
    mdl._interrupted_success = False

    def callback(model_cb: gp.Model, where):
        assert where != gp.GRB.Callback.MULTIOBJ
        if where == gp.GRB.Callback.MIPSOL:
            if model_cb._oracle(model_cb.cbGetSolution(s)):
                mdl._interrupted_success = True
                model_cb.terminate()

    logging.info(f'Starting recovery of an entry. Will timeout after {timeout}s. Current datetime is: {datetime.now()}')
    mdl.optimize(callback)

    assert mdl.status == gp.GRB.TIME_LIMIT or mdl.status == gp.GRB.INTERRUPTED or mdl.status == gp.GRB.OPTIMAL
    assert mdl.status != gp.GRB.INTERRUPTED or mdl._interrupted_success  # interrupted -> interrupted_success

    result = {}
    if mdl.status == gp.GRB.OPTIMAL or mdl.status == gp.GRB.INTERRUPTED:
        result[ResultKeys.DURATION] = mdl.Runtime
        result[ResultKeys.FAILURE] = False
        result[ResultKeys.EQUATIONS_USED] = mdl.ObjVal

        logging.info(f'We used {mdl.ObjVal} zero coefficients in y to recover the secret key.')
        logging.info(f'Recovery took {mdl.Runtime} seconds.')
    else:
        logging.error(f'Gurobi failed: "{GUROBI_STATUS_CODE_TO_STRING[mdl.status]}"')

        result[ResultKeys.FAILURE] = True
        result[ResultKeys.FAILURE_REASON] = GUROBI_STATUS_CODE_TO_STRING[mdl.status]

    x_recovered = x.X
    s_recovered = s.X
    mdl.dispose()
    return (None, None, result) if result[ResultKeys.FAILURE] else (np.around(s_recovered).astype(params.dtype), np.around(x_recovered).astype(np.bool), result)


def recover_s_1(debug_sigs: List[DebugSignature], s: npt.ArrayLike, params: Parameters, total_faults, timeout: float, threshold: int, no_attack: bool = False):
    s_1_entry_result_dict = {ResultKeys.FAULTED_COEFFS: total_faults}

    sigs = [debug_sig.signature for debug_sig in debug_sigs]

    A = np.vstack([sig.c_matrix for sig in sigs])
    b = np.concatenate([sig.z_1 for sig in sigs])
    y = np.concatenate([debug_sig.y for debug_sig in debug_sigs])

    to_keep_mask = np.absolute(b) <= threshold
    A_filtered, b_filtered, y_filtered = A[to_keep_mask], b[to_keep_mask], y[to_keep_mask]
    y_is_zero_after_filter_mask = y_filtered == 0

    s_1_entry_result_dict[ResultKeys.TOTAL_EQUATIONS] = A.shape[0]
    s_1_entry_result_dict[ResultKeys.FILTERED_EQUATIONS] = A_filtered.shape[0]

    logging.info(f'Before filter: {A.shape[0]}; after filter: {A_filtered.shape[0]}; delta: {A.shape[0] - A_filtered.shape[0]}; threshold: {threshold}')
    logging.info(f'Total filtered: {A_filtered.shape[0]}; true: {total_faults}; false: {A_filtered.shape[0] - total_faults}; false-positive rate: {(A_filtered.shape[0] - total_faults) / A_filtered.shape[0] * 100}%; probability: {hypergeom(A_filtered.shape[0], total_faults, params.n).pmf(params.n)}')

    def oracle(s_1_candidate: npt.ArrayLike) -> bool:
        s_1_candidate_rounded = np.around(s_1_candidate).astype(params.dtype)
        if np.all(s_1_candidate_rounded == s):
            logging.info('Candidate is correct with same sign')
            return True
        if np.all(s_1_candidate_rounded == -s):
            logging.info('Candidate is correct with different sign')
            return True
        return False

    recovered_s, x, ilp_result_dict = solve_max_rows_simplified(A_filtered, b_filtered, params, timeout=timeout, oracle=oracle)

    s_1_entry_result_dict = s_1_entry_result_dict | ilp_result_dict

    if recovered_s is not None:  # implies that x is not None
        if np.all(y_is_zero_after_filter_mask == x):
            logging.debug('All zero entries were found and non of the found was classified incorrectly!')
        if np.any(x & np.logical_not(y_is_zero_after_filter_mask)):
            logging.warning('The ILP classified non zero entries as zero.')

    if not s_1_entry_result_dict[ResultKeys.FAILURE] and not np.all(s == recovered_s):
        logging.warning('Recovery of s_1 entry failed because the result is wrong.')
        s_1_entry_result_dict[ResultKeys.FAILURE] = True
        s_1_entry_result_dict[ResultKeys.FAILURE_REASON] = 'WRONG_RESULT'

        different_pos = s != recovered_s
        assert False
    elif not s_1_entry_result_dict[ResultKeys.FAILURE]:
        logging.info('Successfully recovered s_1 entry!')
    else:
        logging.info('Failed to recover s_1 entry!')

    return recovered_s, s_1_entry_result_dict
