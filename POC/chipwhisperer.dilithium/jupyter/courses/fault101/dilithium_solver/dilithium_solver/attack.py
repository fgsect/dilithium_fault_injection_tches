#!/usr/bin/env python3

import logging
from typing import List, Tuple

from parameters import Parameters
from recover_s_1_entry import recover_s_1_entry
from result import ResultKeys
from signature import Attack, gen_sigs_until_success
from multiprocessing import Pool


def attack_success_rate(num_attacks: int, max_num_failures: int, m: int, timeout_per_entry: float, threshold: int, params: Parameters, notion_of_success: int = None) -> Tuple[List[dict], bool]:
    num_failures = 0
    num_successes = 0
    results = []
    for _ in range(num_attacks):
        logging.debug(f'Performing an attack!')
        result = attack_enough_sigs(m, timeout_per_entry, threshold, params, notion_of_success=notion_of_success)
        results.append(result)

        failure = any([entry_result[ResultKeys.FAILURE] for entry_result in result[ResultKeys.ENTRY_RESULTS]])

        num_failures += failure
        num_successes += not failure

        if num_failures > max_num_failures:  # we break because we were not able to reach the success rate
            break
        if num_successes >= num_attacks - max_num_failures:  # we break because we achieved the needed success rate
            break

    return results, num_failures <= max_num_failures


def attack_enough_sigs(m: int, timeout_per_entry: float, threshold: int, params: Parameters, notion_of_success: int = None) -> dict:
    logging.info(f'We will have m={m} non-zero entries. Meaning we will have {params.n * params.l - m} zero entries.')

    attack_data = gen_sigs_until_success(m, threshold, params, notion_of_success=notion_of_success)

    logging.info(f'We will use {len(attack_data.debug_signatures)} signatures.')

    result_dict = attack(attack_data, timeout_per_entry, threshold, params) | {
                       ResultKeys.M: m,
                       ResultKeys.NUM_SIGNATURES: len(attack_data.debug_signatures),
                       ResultKeys.NIST_PARAM_LEVEL: params.nist_security_level,
                       ResultKeys.THRESHOLD: threshold,
                       ResultKeys.NOTION_OF_SUCCESS: notion_of_success,
                       ResultKeys.TIMEOUT_LIMIT: timeout_per_entry
    }

    logging.debug(ResultKeys.results_to_json_string(result_dict))
    return result_dict


def attack(attack_data: Attack, time_limit: float, threshold: int, params: Parameters) -> dict:
    if threshold is None:
        threshold = params.beta

    debug_signatures, s_1, true_equations_per_entry, false_positive_rate_per_entry = attack_data
    sigs = [ds.signature for ds in debug_signatures]

    entry_results = []
    remaining_runtime = time_limit
    for i in range(params.l):
        s_recovered_array, res_dict = recover_s_1_entry(sigs, i, s_1, params, true_equations_per_entry[i], remaining_runtime, threshold)
        remaining_runtime = max(0, remaining_runtime - res_dict[ResultKeys.DURATION])
        entry_results.append(res_dict)

    if all([not entry_result[ResultKeys.FAILURE] for entry_result in entry_results]):
        logging.info('Successful recovery!')
    else:
        logging.warning('Failed recovery!')

    return {ResultKeys.ENTRY_RESULTS: entry_results}

