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

    time_limit = 30
    nist_security_level = 2
    threads = 40
    params = Parameters.get_nist_security_level(nist_security_level, threads=threads)
    notion_of_success = params.n

    datetime_str = str(datetime.now()).replace(':', '.')
    logging.getLogger().addHandler(logging.FileHandler(f'logs/{datetime_str}'))
    for f in range(1, 10 + 1):
        m = params.n * params.l - f
        for threshold in range(1, params.beta + 1):
            r = attack_enough_sigs(m, time_limit, threshold, params, notion_of_success=notion_of_success)
            successful = all(not [entry[ResultKeys.FAILURE]] for entry in r[ResultKeys.ENTRY_RESULTS])

            if not successful:
                logging.info(f'm = {m}; threshold = {threshold} did not succeed!')
            else:
                logging.info(f'm = {m}; threshold = {threshold} succeeded!')


if __name__ == '__main__':
    main()
