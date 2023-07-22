#!/usr/bin/env python3
import logging
from datetime import datetime
from parameters import Parameters
from attack import attack_success_rate
from pathlib import Path

from result import ResultKeys


def main() -> None:
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger('gurobipy.gurobipy').setLevel(logging.WARN)

    datetime_str = str(datetime.now()).replace(':', '.')
    Path('logs').mkdir(exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(f'logs/{datetime_str}'))

    logging.debug('The script is running!')

    attacks_per_m = 3
    max_failed_attacks = 1
    parameter_set = 1
    threads = 40
    params = Parameters(parameter_set, threads)
    threshold = params.beta
    notion_of_success = params.n

    for m in range(0, params.n):
        while True:
            r, successful = attack_success_rate(attacks_per_m, max_failed_attacks, m, 5 * 60, threshold, params, notion_of_success=notion_of_success)

            min_total_nonfiltered_true_equations = min([entry[ResultKeys.FAULTED_COEFFS] for res in r for entry in res[ResultKeys.ENTRY_RESULTS]])
            assert min_total_nonfiltered_true_equations >= notion_of_success
            if not successful:
                logging.info(f'min_true: {min_total_nonfiltered_true_equations}; m = {m}; notion_of_success = {notion_of_success} did not succeed, trying again with {min_total_nonfiltered_true_equations + 1}')
                notion_of_success = min_total_nonfiltered_true_equations + 1
            else:
                logging.info(f'min_true: {min_total_nonfiltered_true_equations}; m = {m}; notion_of_success = {notion_of_success} succeeded!')
                break


if __name__ == '__main__':
    main()
