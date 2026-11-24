"""Minimal non-GUI smoke check using explicitly synthetic evidence."""

import json

from experiment_runner import run_comparison
from affective_v03.scenarios import demonstration_scenarios


if __name__ == "__main__":
    observation = next(iter(demonstration_scenarios().values()))
    print(json.dumps(run_comparison(observation), ensure_ascii=False, indent=2))
