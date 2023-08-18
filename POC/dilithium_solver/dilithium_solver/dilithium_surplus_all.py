#!/usr/bin/env python3
import logging
import math
from datetime import datetime

from parameters import Parameters
from attack import attack_enough_sigs
from result import ResultKeys


def main() -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger('gurobipy.gurobipy').setLevel(logging.WARN)

    logging.debug('The script is running!')

    time_limit = 30# * 5
    attacks_per_m = 3
    max_failed_attacks = 1
    nist_security_level = 2
    threads =40
    params = Parameters.get_nist_security_level(nist_security_level, threads=threads)
    threshold = params.beta
    # notion_of_success = params.n
    notion_of_success_max = 2 * params.n

    datetime_str = str(datetime.now()).replace(':', '.')
    logging.getLogger().addHandler(logging.FileHandler(f'logs/{datetime_str}'))
    for f in range(1, 10 + 1):
        m = params.n * params.l - f
        for notion_of_success in range(params.n, notion_of_success_max + 1,):
            r = attack_enough_sigs(m, time_limit, threshold, params, notion_of_success=notion_of_success)
            successful = all(not [entry[ResultKeys.FAILURE]] for entry in r[ResultKeys.ENTRY_RESULTS])

            if not successful:
                logging.info(f'm = {m}; notion_of_success = {notion_of_success} did not succeed!')
            else:
                logging.info(f'm = {m}; notion_of_success = {notion_of_success} succeeded!')


if __name__ == '__main__':
    main()
