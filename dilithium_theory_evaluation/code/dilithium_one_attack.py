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

    attacks_per_m = 3
    max_failed_attacks = 1
    nist_security_level = 2
    threads = 40
    params = Parameters.get_nist_security_level(nist_security_level, threads=threads)
    threshold = params.beta
    notion_of_success = 2*params.n # How many _correct_ equations (for each secret key polynomial) do we collect before we launch the attack?
    datetime_str = str(datetime.now()).replace(':', '.')
    print("n is ", params.n, "l is ", params.l)
    #m measures how many coefficients are NOT faulted
    #That means we fault n*l -m coefficients
    #Setting m to params.n*params.l -2 will fault exactly TWO coefficient per signature
    #The script will inform the user if the attack succeeded
    for m in range(params.n*params.l - 2, params.n * params.l -1): 
        while True:
            r, successful = attack_success_rate(attacks_per_m, max_failed_attacks, m, 5 * 60, threshold, params, notion_of_success=notion_of_success)

            if not successful:
                notion_of_success_old = notion_of_success
                notion_of_success += 1
                logging.info(f'm = {m}; notion_of_success = {notion_of_success_old} did not succeed, trying again with {notion_of_success}')
            else:
                logging.info(f'm = {m}; notion_of_success = {notion_of_success} succeeded!')
                break


if __name__ == '__main__':
    main()
