#!/usr/bin/env python3
import logging
from datetime import datetime

from parameters import Parameters
from attack import attack_success_rate


def main() -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger('gurobipy.gurobipy').setLevel(logging.WARN)

    logging.debug('The script is running!')

    attacks_per_m = 3
    max_failed_attacks = 1
    nist_security_level = 5
    threads = 40
    params = Parameters.get_nist_security_level(nist_security_level, threads=threads)
    threshold = params.beta
    notion_of_success = params.n

    datetime_str = str(datetime.now()).replace(':', '.')
    logging.getLogger().addHandler(logging.FileHandler(f'logs/{datetime_str}'))
    for m in range(params.n * params.l - 10, params.n * params.l):  # 10 just to be sure
        while True:
            r, successful = attack_success_rate(attacks_per_m, max_failed_attacks, m, 5 * 60, threshold, params, notion_of_success=notion_of_success)

            if not successful:
                if threshold == 1:
                    logging.info(f'We failed with a threshold of 1, thus we can not continue further as threshold 0 not solveable with the ILP.')
                    exit()
                threshold_old = threshold
                threshold -= 1
                logging.info(f'm = {m}; threshold = {threshold_old} did not succeed, trying again with {threshold}')
            else:
                logging.info(f'm = {m}; threshold = {threshold} succeeded!')
                break


if __name__ == '__main__':
    main()
