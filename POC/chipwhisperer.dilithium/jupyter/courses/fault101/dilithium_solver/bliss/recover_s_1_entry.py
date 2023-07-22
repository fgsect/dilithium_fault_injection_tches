import logging
from typing import List, Tuple, Optional
import numpy as np
import gurobipy as gp
from datetime import datetime
from parameters import Parameters
import numpy.typing as npt
from result import ResultKeys
from signature import DebugSignature, calculate_z_1_matrix
from scipy.stats import hypergeom


def solve_max_rows_simplified(Cs: list[npt.ArrayLike], zs: list[npt.ArrayLike], params: Parameters, equations_per_signature: list[int], time_limit: float, check_is_success) -> Tuple[Optional[npt.ArrayLike], Optional[npt.ArrayLike], Optional[npt.ArrayLike], dict]:
    model = gp.Model()

    C = np.vstack(Cs)
    z = np.concatenate(zs)

    x = model.addMVar(shape=np.shape(z), vtype=gp.GRB.BINARY)
    s = model.addMVar(shape=params.n, lb=params.s_1_range.start, ub=params.s_1_range.stop - 1, vtype=gp.GRB.INTEGER)
    b = model.addMVar(shape=len(equations_per_signature), vtype=gp.GRB.BINARY)  # 1 means 1, 0 means -1

    model.setObjective(x.sum(), gp.GRB.MAXIMIZE)

    z_1_matrix = calculate_z_1_matrix(zs, params)
    z_expression = z_1_matrix @ b - z  # this is the z vector but with (hopefully) correct signs
    model.addConstr(z_expression - C @ s <= params.max_diff * (1 - x))
    model.addConstr(z_expression - C @ s >= -params.max_diff * (1 - x))

    model.params.TimeLimit = time_limit
    if params.threads is not None:
        model.params.Threads = params.threads

    logging.info(f'Starting recovery of an entry. Will timeout after {time_limit}s. Current datetime is: {datetime.now()}')

    model._check_is_success = check_is_success
    model._interrupted_success = False

    def callback(model_cb: gp.Model, where):
        assert where != gp.GRB.Callback.MULTIOBJ
        if where == gp.GRB.Callback.MIPSOL:
            if model_cb._check_is_success(model_cb.cbGetSolution(s)):
                model._interrupted_success = True
                model_cb.terminate()
            else:
                logging.debug('Candidate is not correct.')

    model.optimize(callback)

    logging.info(f'Gurobi finish status: "{ResultKeys.GUROBIPY()[model.status]}"')
    assert model.status == gp.GRB.TIME_LIMIT or model.status == gp.GRB.INTERRUPTED or model.status == gp.GRB.OPTIMAL
    assert model.status != gp.GRB.INTERRUPTED or model._interrupted_success  # interrupted -> interrupted_success

    result = {
        ResultKeys.RUNTIME: model.Runtime,
        ResultKeys.STATUS: model.status,
        ResultKeys.EQUATIONS_USED: model.ObjVal
    }

    s_recovered, x_recovered, b_recovered = np.around(s.X).astype(params.dtype), np.around(x.X).astype(np.bool), np.around(b.X).astype(np.bool)

    if check_is_success(s_recovered):
        logging.info(f'We used {model.ObjVal} zero coefficients in y to recover the secret key.')
        logging.info(f'Recovery took {model.Runtime} seconds.')
    else:
        logging.error(f'Gurobi failed: "{ResultKeys.GUROBIPY()[model.status]}"')

    model.dispose()
    return s_recovered, x_recovered, b_recovered, result


def recover_s_1(debug_signatures: List[DebugSignature], s_1: npt.ArrayLike, params: Parameters, total_faults: int, time_limit: float, threshold: int):
    eqs_per_signature = [np.sum(np.abs(debug_signature.signature.z_1) <= threshold) for debug_signature in debug_signatures]

    Cs_filtered = []
    z_1s_filtered = []
    # y_1s = []
    for debug_signature in debug_signatures:
        to_keep_mask = np.absolute(debug_signature.signature.z_1) <= threshold
        Cs_filtered.append(debug_signature.signature.c_matrix[to_keep_mask])
        z_1s_filtered.append(debug_signature.signature.z_1[to_keep_mask])
        # y_1s.append(debug_signature.signature.z_1[to_keep_mask])

    C = np.vstack([debug_signature.signature.c_matrix for debug_signature in debug_signatures])
    z_1 = np.concatenate([debug_signature.signature.z_1 for debug_signature in debug_signatures])
    y_1 = np.concatenate([debug_signature.y_1 for debug_signature in debug_signatures])

    to_keep_mask = np.absolute(z_1) <= threshold
    C_filtered, z_1_filtered, y_1_filtered = C[to_keep_mask], z_1[to_keep_mask], y_1[to_keep_mask]

    result = {
        ResultKeys.TOTAL_EQUATIONS: C.shape[0],
        ResultKeys.FILTERED_EQUATIONS: C_filtered.shape[0]
    }

    logging.info(f'Before filter: {C.shape[0]}; after filter: {C_filtered.shape[0]}; delta: {C.shape[0] - C_filtered.shape[0]}; threshold: {threshold}')
    logging.info(f'Total filtered: {C_filtered.shape[0]}; true: {total_faults}; false: {C_filtered.shape[0] - total_faults}; false-positive rate: {(C_filtered.shape[0] - total_faults) / C_filtered.shape[0] * 100}%; probability: {hypergeom(C_filtered.shape[0], total_faults, params.n).pmf(params.n)}')

    def check_is_correct(s_1_candidate: npt.ArrayLike) -> bool:
        s_1_candidate_rounded = np.around(s_1_candidate).astype(params.dtype)
        if np.all(s_1_candidate_rounded == s_1):
            logging.info('Candidate is correct with same sign')
            return True
        if np.all(s_1_candidate_rounded == -s_1):
            logging.info('Candidate is correct with different sign')
            return True
        return False

    s_1_recovered, x, b, result_ilp = solve_max_rows_simplified(Cs_filtered, z_1s_filtered, params, eqs_per_signature, time_limit, check_is_correct)
    logging.debug(f's_1_recovered is None? {s_1_recovered is None}; x is None? {x is None}')

    b_correct = np.array([0 if ds.sign == -1 else ds.sign for ds in debug_signatures], dtype=np.bool)

    if s_1_recovered is not None and x is not None and b is not None:
        zero_coefficients_found = x & (y_1_filtered == 0)
        incorrectly_classified_as_zero = x & np.logical_not(y_1_filtered == 0)
        incorrect_s_1_entries = s_1_recovered != s_1
        incorrect_b_entries = b == b_correct
        logging.info(f'num_incorrect_b_entries: {np.sum(incorrect_b_entries)}')
        if np.sum(zero_coefficients_found) == total_faults and np.sum(incorrectly_classified_as_zero) == 0:
            logging.debug('All zero entries were found and non of the found was classified incorrectly!')
            assert np.all(s_1_recovered == s_1) or np.all(s_1_recovered == -s_1)
        else:
            if np.sum(zero_coefficients_found) < total_faults and np.sum(incorrectly_classified_as_zero) == 0:
                logging.debug('Not all zero entries were found and non of the found were classified incorrectly!')
            if np.sum(incorrectly_classified_as_zero) > 0:
                logging.debug('Not all zero entries were found and some of the found were classified incorrectly!')

            logging.warning(f'Num correct classified: {np.sum(zero_coefficients_found)}')
            logging.warning(f'Num incorrect classified: {np.sum(incorrectly_classified_as_zero)}')
            logging.warning(f'Num incorrect s_1 coefficients {np.sum(incorrect_s_1_entries)}')

            C_found, z_1_found = C_filtered[zero_coefficients_found], z_1_filtered[zero_coefficients_found]
            _, _, rank_correctly_classified, _ = np.linalg.lstsq(C_found, z_1_found, rcond=None)
            logging.warning(f'Rank of correctly classified equations: {rank_correctly_classified}')
            # assert rank_correctly_classified < params.n  # can no longer assert this because signs might be incorrect

            C_candidate, z_1_candidate = C_filtered[x], z_1_filtered[x]
            s_1_recovered_again, _, candidate_rank, _ = np.linalg.lstsq(C_candidate, z_1_candidate, rcond=None)
            logging.debug(f'Rank of candidate: {candidate_rank}')
            if candidate_rank == params.n and False: # can no longer assert this because of signs
                assert np.all(np.around(s_1_recovered_again).astype(params.dtype) == s_1_recovered)
                guess_correct = np.all(np.around(s_1_recovered_again).astype(params.dtype) == s_1_recovered)
                logging.debug(f's_1_recovered_again and s_1_recovered are same: {guess_correct}; rank of candidate: {candidate_rank}')

    success = s_1_recovered is not None and (np.all(s_1_recovered == s_1) or np.all(s_1_recovered == -s_1))
    result[ResultKeys.FAILURE] = not success
    if result[ResultKeys.FAILURE]:
        logging.info('Failed recovery!')
    else:
        logging.info('Successful recovery!')

    return s_1_recovered, result | result_ilp
