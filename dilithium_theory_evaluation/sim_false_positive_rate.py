import numpy as np
import pandas as pd
from signature import gen_sigs_until_success
from parameters import Parameters

pd.set_option('display.max_columns', None)


# res = {
#     2: {
#         (2, 35): None,
#         (3, 73): None,
#         (4, 78): None,
#     },
#     3: {
#         (2, 45): None,
#         (3, 178): None,
#         (4, 194): None,
#     },
#     5: {
#         (1, 13): None,
#         (2, 103): None,
#         (3, 120): None,
#         (4, 120): None,
#     }
# }

res = {
    2: {
        (2, 1): None,
        (3, 1): None,
        (4, 1): None,
    },
    3: {
        (2, 1): None,
        (3, 1): None,
        (4, 1): None,
    },
    5: {
        (1, 1): None,
        (2, 1): None,
        (3, 1): None,
        (4, 1): None,
    }
}


def get_false_positive_rate(nist_security_level: int, f: int, threshold: int, iteration_count=10) -> float:
    total = 0
    for i in range(iteration_count):
        params = Parameters.get_nist_security_level(nist_security_level)
        attack = gen_sigs_until_success(params.l * params.n - f, threshold, params)
        mean_fpr = np.mean(attack.false_positive_rate_per_entry)
        total += mean_fpr
    return total / iteration_count


def main() -> None:
    columns = ['f']
    for nist_security_level in res.keys():
        columns += [f'l{nist_security_level}_nist_security_level',
                    f'l{nist_security_level}_threshold', f'l{nist_security_level}_false_positive_rate']

    dfs = []
    for nist_security_level in res.keys():
        df_level = pd.DataFrame(columns=['f', f'l{nist_security_level}_nist_security_level', f'l{nist_security_level}_threshold', f'l{nist_security_level}_false_positive_rate']).set_index('f')
        for f, threshold in res[nist_security_level]:
            false_positive_rate = get_false_positive_rate(nist_security_level, f, threshold, iteration_count=10)
            print(f'l={nist_security_level};f={f};threshold={threshold};false_positive_rate={false_positive_rate}')
            new_row = pd.DataFrame([{f'l{nist_security_level}_nist_security_level': nist_security_level,
                                     f'f': f,
                                     f'l{nist_security_level}_threshold': threshold,
                                     f'l{nist_security_level}_false_positive_rate': false_positive_rate
                                     }]).set_index(f'f')
            df_level = pd.concat([df_level, new_row])
        dfs += [df_level]
    all_levels = dfs[0].join(dfs[1:], how='outer').sort_index()
    print(all_levels)
    all_levels.to_json('notebooks/simulated_false_positive_rates_min.json')


if __name__ == '__main__':
    main()
