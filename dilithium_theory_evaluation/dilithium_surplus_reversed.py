#!/usr/bin/env python3
import logging
from datetime import datetime

from parameters import Parameters
from result import ResultKeys
from attack import attack_success_rate


def main() -> None:
    logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger('gurobipy.gurobipy').setLevel(logging.WARN)

    logging.debug('The script is running!')

    time_limit = 60 * 5
    attacks_per_m = 3
    max_failed_attacks = 1
    nist_security_level = 2
    threads = 40
    params = Parameters.get_nist_security_level(nist_security_level, threads=threads)
    threshold = params.beta
    notion_of_success = 2 * params.n

    datetime_str = str(datetime.now()).replace(':', '.')
    logging.getLogger().addHandler(logging.FileHandler(f'logs/{datetime_str}'))
    for f in range(1, 4 + 1):
        m = params.n * params.l - f
        while True:
            r, successful = attack_success_rate(attacks_per_m, max_failed_attacks, m, time_limit, threshold, params, notion_of_success=notion_of_success)

            if not successful:
                logging.info(f'm = {m}; notion_of_success = {notion_of_success} did not succeed, trying again with increased f = {f + 1} (decreased m = {m - 1})')

                if notion_of_success == params.n:
                    logging.info(
                        f'We failed with a notion_of_success of n = {params.n}. f = {f} (m = {m}) seems to be the best we can do with notion_of_success.')
                    exit()

                break
            else:
                logging.info(f'm = {m}; notion_of_success = {notion_of_success} succeeded! Will it work with notion_of_success = {notion_of_success - 1} too?')
                notion_of_success -= 1

                if notion_of_success == params.n:
                    logging.info(
                        f'We failed with a notion_of_success of n = {params.n}. f = {f} (m = {m}) seems to be the best we can do with threshold.')
                    exit()


if __name__ == '__main__':
    main()
