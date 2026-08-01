import argparse

import pyperf

from fnn.performance.scenarios import BENCHMARK_SCENARIOS, BenchmarkScenario


def _selected_scenarios(names: list[str] | None) -> list[BenchmarkScenario]:
    if not names:
        return list(BENCHMARK_SCENARIOS.values())
    unknown = sorted(set(names) - set(BENCHMARK_SCENARIOS))
    if unknown:
        raise SystemExit(f"Unknown benchmark scenario(s): {', '.join(unknown)}")
    return [BENCHMARK_SCENARIOS[name] for name in names]


def _list_scenarios() -> None:
    for scenario in BENCHMARK_SCENARIOS.values():
        print(f"{scenario.name}: {scenario.description}")


def _add_cmdline_args(cmd: list[str], args: argparse.Namespace) -> None:
    for scenario in args.scenarios or []:
        cmd.extend(("--scenario", scenario))


def main() -> None:
    runner = pyperf.Runner(
        program_args=("-m", "fnn.performance.runner"),
        add_cmdline_args=_add_cmdline_args,
    )
    runner.argparser.add_argument(
        "--scenario",
        action="append",
        choices=sorted(BENCHMARK_SCENARIOS),
        dest="scenarios",
        help="Benchmark scenario to run. Can be passed more than once.",
    )
    runner.argparser.add_argument(
        "--list",
        action="store_true",
        help="List available scenarios and exit.",
    )
    args = runner.parse_args()
    if args.list:
        _list_scenarios()
        return
    scenarios = _selected_scenarios(args.scenarios)
    for scenario in scenarios:
        runner.bench_func(scenario.name, scenario.make_callable())


if __name__ == "__main__":
    main()
