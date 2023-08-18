import logging
from typing import List
import numpy as np
import gurobipy as gp
from datetime import datetime
from .parameters import Parameters
import numpy.typing as npt
from .result import ResultKeys
from .signature import Signature
from scipy.stats import hypergeom


def solve_max_rows_simplified(A, b, params: Parameters, timeout: float, oracle):
    GUROBI_STATUS_CODE_TO_STRING = {gp.StatusConstClass.__dict__[k]: k for k in gp.StatusConstClass.__dict__.keys() if
                                    'A' <= k[0] <= 'Z'}

    with gp.Env() as env, gp.Model(env=env) as model:

        x = model.addMVar(shape=np.shape(b), vtype=gp.GRB.BINARY)
        s = model.addMVar(shape=np.shape(A)[1], lb=params.s_1_range.start, ub=params.s_1_range.stop-1, vtype=gp.GRB.INTEGER)

        model.setObjective(x.sum(), gp.GRB.MAXIMIZE)

        model.addConstr(A @ s - b <= params.max_diff * (1 - x))
        model.addConstr(A @ s - b >= -params.max_diff * (1 - x))

        model.params.TimeLimit = timeout
        if params.threads is not None:
            model.params.Threads = params.threads

        logging.info(f'Starting recovery of an entry. Will timeout after {timeout}s. Current datetime is: {datetime.now()}')

        model._oracle = oracle
        model._interrupted_success = False

        def callback(model_cb: gp.Model, where):
            assert where != gp.GRB.Callback.MULTIOBJ
            if where == gp.GRB.Callback.MIPSOL:
                if model_cb._oracle(model_cb.cbGetSolution(s)):
                    model._interrupted_success = True
                    model_cb.terminate()

        model.optimize(callback)

        assert model.status == gp.GRB.TIME_LIMIT or model.status == gp.GRB.INTERRUPTED or model.status == gp.GRB.OPTIMAL
        assert model.status != gp.GRB.INTERRUPTED or model._interrupted_success  # interrupted -> interrupted_success

        logging.debug(f'Gurobi result status: "{GUROBI_STATUS_CODE_TO_STRING[model.status]}"')

        result = {ResultKeys.DURATION: model.Runtime}
        if model.status == gp.GRB.OPTIMAL or model.status == gp.GRB.INTERRUPTED:
            result[ResultKeys.FAILURE] = False
            result[ResultKeys.EQUATIONS_USED] = model.ObjVal

            logging.info(f'We used {model.ObjVal} zero coefficients in y to recover the secret key.')
            logging.info(f'Recovery took {model.Runtime} seconds.')
        else:
            logging.error(f'Gurobi failed: "{GUROBI_STATUS_CODE_TO_STRING[model.status]}"')

            result[ResultKeys.FAILURE] = True
            result[ResultKeys.FAILURE_REASON] = GUROBI_STATUS_CODE_TO_STRING[model.status]

        s_recovered = s.X
    return None if result[ResultKeys.FAILURE] else s_recovered, result


def recover_s_1_entry(sigs: List[Signature], s_1_entry_index: int, s_1: npt.ArrayLike, params: Parameters, num_zero_coefficients, timeout: float, threshold: int):
    s_1_entry_result_dict = {ResultKeys.FAULTED_COEFFS: num_zero_coefficients}

    A = np.vstack([sig.c_matrix for sig in sigs])
    b = np.concatenate([sig.z[s_1_entry_index] for sig in sigs])

    to_keep_mask = np.absolute(b) <= threshold
    A_filtered, b_filtered = A[to_keep_mask], b[to_keep_mask]

    s_1_entry_result_dict[ResultKeys.TOTAL_EQUATIONS] = A.shape[0]
    s_1_entry_result_dict[ResultKeys.FILTERED_EQUATIONS] = A_filtered.shape[0]

    logging.info(f'Before filter: {A.shape[0]}; after filter: {A_filtered.shape[0]}; delta: {A.shape[0] - A_filtered.shape[0]}; threshold: {threshold}')
    logging.info(f'Total filtered: {A_filtered.shape[0]}; true: {num_zero_coefficients}; false: {A_filtered.shape[0] - num_zero_coefficients}; false-positive rate: {(A_filtered.shape[0] - num_zero_coefficients) / A_filtered.shape[0] * 100}%; probability: {hypergeom(A_filtered.shape[0], num_zero_coefficients, params.n).pmf(params.n)}')

    def oracle(s_1_i_candidate: npt.ArrayLike) -> bool:
        s_1_i_candidate_rounded = np.around(s_1_i_candidate).astype(params.dtype)
        if np.all(s_1_i_candidate_rounded == s_1[s_1_entry_index]):
            logging.info('Candidate is correct with same sign')
            return True
        if np.all(s_1_i_candidate_rounded == -s_1[s_1_entry_index]):
            logging.info('Candidate is correct with different sign')
            return True
        return False

    recovered_s, ilp_result_dict = solve_max_rows_simplified(A_filtered, b_filtered, params, timeout=timeout, oracle=oracle)

    s_1_entry_result_dict = {**s_1_entry_result_dict, **ilp_result_dict}

    if not s_1_entry_result_dict[ResultKeys.FAILURE] and not np.array_equal(s_1[s_1_entry_index, :], recovered_s):
        logging.warning('Recovery of s_1 entry failed because the result is wrong.')
        s_1_entry_result_dict[ResultKeys.FAILURE] = True
        s_1_entry_result_dict[ResultKeys.FAILURE_REASON] = 'WRONG_RESULT'
    elif not s_1_entry_result_dict[ResultKeys.FAILURE]:
        logging.info('Successfully recovered s_1 entry!')
    else:
        logging.info('Failed to recover s_1 entry!')

    return recovered_s, s_1_entry_result_dict

