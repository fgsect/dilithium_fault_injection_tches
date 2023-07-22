import json
from collections import defaultdict


def load_results(paths: [str]) -> list[dict]:
    results = []
    for path in paths:
        with open(path) as file:
            lines = file.readlines()
        for line in lines:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return [{'attack_no': i} | result for i, result in enumerate(results)]


def get_structured_results(all_results):
    results_structured = defaultdict(lambda: [])
    for result in all_results:
        results_structured[(result['m'], result['notion_of_success'])].append(result)

    for (m, nos) in results_structured:
        num_failures = 0
        num_successes = 0
        run = 0
        for i in range(len(results_structured[(m, nos)])):
            results_structured[(m, nos)][i] = {'run_no': run} | results_structured[(m, nos)][i]
            if results_structured[(m, nos)][i]['failure']:
                num_failures += 1
            else:
                num_successes += 1
            run += 1
            if num_successes >= 2 or num_failures >= 2:
                num_failures = num_successes = run = 0

    return results_structured


def main() -> None:
    all_results = load_results(['logs/l3_surplus_fast.txt',
                                'logs/l3_surplus_continued.txt',
                                'logs/l3_surplus_continued2.txt',
                                'logs/l3_surplus_continued3.txt',
                                'logs/l3_surplus_continued4.txt'])

    for (m, nos), results in get_structured_results(all_results).items():
        if len(results) > 3:
            print(m, nos, len(results))
            for result in results:
                print(result)
            print()

    # for some reason m = 375 and nos = 823 was attempted two times,
    # the first one failed and the second did not, to be consistent we remove the first attempt
    all_results = [result for result in all_results if not (784 <= result['attack_no'] <= 785)]

    # for some reason m = 358 and nos = 771 was attempted three times,
    # the first two failed and the third did not, to be consistent we remove the first two attempts
    all_results = [result for result in all_results if not (741 <= result['attack_no'] <= 746)]

    # we did not manage to break m = 376, so we can remove all the attempts for m = 376
    all_results = [result for result in all_results if result['m'] != 376]

    print('Checking again ...')
    for (m, nos), results in get_structured_results(all_results).items():
        if len(results) > 3:
            print(m, nos, len(results))
            for result in results:
                print(result)
            print()
    print('Checks done!')

    with open('logs/l3_surplus_fast_final.txt', 'w') as file:
        for result in all_results:
            # json.dump(result, file)
            file.write(json.dumps(result))
            file.write('\n')
    print('Done writing!')


if __name__ == '__main__':
    main()
