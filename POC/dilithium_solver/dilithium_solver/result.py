import json
from enum import Enum
from operator import attrgetter
from typing import List, Union
import numpy as np


class ResultKeys(Enum):
    @staticmethod
    def results_to_json(results: Union[dict, List[dict]]) -> Union[dict, List[dict]]:
        def convert_keys(obj, convert=attrgetter('value')):
            """src: https://stackoverflow.com/questions/43854335/encoding-python-enum-to-json (first answer)"""
            if isinstance(obj, list):
                return [convert_keys(i, convert) for i in obj]
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if not isinstance(obj, dict):
                return obj
            return {convert(k): convert_keys(v, convert) for k, v in obj.items()}

        return convert_keys(results)

    @staticmethod
    def results_to_json_string(results: dict) -> str:
        replaced_dict = ResultKeys.results_to_json(results)
        return json.dumps(replaced_dict)

    FAILURE = 'failure'
    FAILURE_REASON = 'failure_reason'
    TOTAL_EQUATIONS = 'total_equations'
    FILTERED_EQUATIONS = 'filtered_equations'
    FALSE_POSITIVE_RATE = 'false_positive_rate'
    DURATION = 'duration'
    TIMEOUT_LIMIT = 'timeout_limit'
    EQUATIONS_USED = 'equations_used'
    FAULTED_COEFFS = 'faulted_coeffs'
    TOTAL_DURATION = 'total_duration'
    ENTRY_RESULTS = 'entry_results'
    M = 'm'
    SUCCESS_PROBABILITY = 'success_probability'
    NUM_SIGNATURES = 'num_signatures'
    NIST_PARAM_LEVEL = 'nist_param_level'
    THRESHOLD = 'threshold'
    NOTION_OF_SUCCESS = 'notion_of_success'
