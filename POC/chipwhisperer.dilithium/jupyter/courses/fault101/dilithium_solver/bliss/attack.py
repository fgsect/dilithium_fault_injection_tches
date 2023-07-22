#!/usr/bin/env python3

import logging
from typing import List, Tuple
import numpy as np
from parameters import Parameters
from recover_s_1_entry import recover_s_1
from result import ResultKeys
from signature import Attack, gen_sigs_until_success


def attack_success_rate(num_attacks: int, max_num_failures: int, m: int, timeout_per_entry: float, threshold: int, params: Parameters, notion_of_success: int = None) -> Tuple[List[dict], bool]:
    num_failures = 0
    num_successes = 0
    results = []
    for _ in range(num_attacks):
        result = attack_enough_sigs(m, timeout_per_entry, threshold, params, notion_of_success=notion_of_success)
        results.append(result)

        logging.debug(ResultKeys.results_to_json_string(result))

        num_failures += result[ResultKeys.FAILURE]
        num_successes += not result[ResultKeys.FAILURE]

        if num_failures > max_num_failures:  # we break because we were not able to reach the success rate
            break
        if num_successes >= num_attacks - max_num_failures:  # we break because we achieved the needed success rate
            break

    return results, num_failures <= max_num_failures


def attack_enough_sigs(m: int, timeout_per_entry: float, threshold: int, params: Parameters, notion_of_success: int = None) -> dict:
    logging.info(f'We will have m={m} non-zero entries. Meaning we will have {params.n - m} zero entries.')

    attack_data = gen_sigs_until_success(m, threshold, params, notion_of_success=notion_of_success)

    logging.info(f'We will use {len(attack_data.debug_signatures)} signatures.')

    result = attack(attack_data, timeout_per_entry, threshold, params) | {
                       ResultKeys.M: m,
                       ResultKeys.NUM_SIGNATURES: len(attack_data.debug_signatures),
                       ResultKeys.NIST_PARAM_LEVEL: params.parameter_set,
                       ResultKeys.THRESHOLD: threshold,
                       ResultKeys.NOTION_OF_SUCCESS: notion_of_success,
                       ResultKeys.TIMEOUT_LIMIT: timeout_per_entry,
                       ResultKeys.TOTAL_EQUATIONS: attack_data.total_faults
    }

    return result


def attack(attack_data: Attack, timeout_per_entry: float, threshold: int, params: Parameters) -> dict:
    if threshold is None:
        threshold = params.beta

    debug_signatures, s_1, total_faults = attack_data

    s_recovered, result = recover_s_1(debug_signatures, s_1, params, total_faults, time_limit=timeout_per_entry, threshold=threshold)

    return result
