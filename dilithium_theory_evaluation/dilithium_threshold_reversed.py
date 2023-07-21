#!/usr/bin/env python3
import logging
from datetime import datetime

from parameters import Parameters
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
    threshold = 1
    notion_of_success = params.n

    datetime_str = str(datetime.now()).replace(':', '.')
    logging.getLogger().addHandler(logging.FileHandler(f'logs/{datetime_str}'))
    for f in range(1, 4 + 1):
        m = params.n * params.l - f
        while True:
            r, successful = attack_success_rate(attacks_per_m, max_failed_attacks, m, time_limit, threshold, params, notion_of_success=notion_of_success)

            if not successful:
                logging.info(f'm = {m}; threshold = {threshold} did not succeed, trying again with increased f = {f + 1} (decreased m = {m - 1})')

                if threshold == params.beta:
                    logging.info(
                        f'We failed with a threshold of β = {params.beta}. f = {f} (m = {m}) seems to be the best we can do with threshold.')
                    exit()

                break
            else:
                logging.info(f'm = {m}; threshold = {threshold} succeeded! Will it work with threshold = {threshold + 1} too?')
                threshold += 1

                if threshold == params.beta:
                    logging.info(
                        f'We failed with a threshold of β = {params.beta}. f = {f} (m = {m}) seems to be the best we can do with threshold.')
                    exit()


if __name__ == '__main__':
    main()
