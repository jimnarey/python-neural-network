import argparse
import csv
from pathlib import Path

from fnn.performance.records import RESULTS_DIR


def _latest_result_file() -> Path:
    files = sorted(RESULTS_DIR.glob("*-benchmarks.csv"))
    if not files:
        raise SystemExit(f"No benchmark CSV files found in {RESULTS_DIR}")
    return files[-1]


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as csvfile:
        return list(csv.DictReader(csvfile))


def _get_seconds_unit(value: float) -> tuple[float, str]:
    abs_value = abs(value)
    if abs_value >= 1.0:
        return 1.0, "sec"
    if abs_value >= 1e-3:
        return 1e3, "ms"
    if abs_value >= 1e-6:
        return 1e6, "us"
    if abs_value >= 1e-9:
        return 1e9, "ns"
    return 1.0, "sec"


def _format_seconds(value: float, factor: float, unit: str) -> str:
    return f"{value * factor:.3g} {unit}"


def _print_file(path: Path) -> None:
    rows = _read_rows(path)
    if not rows:
        print(f"{path}: no benchmark rows")
        return
    first_row = rows[0]
    print(
        f"{path.name} "
        f"({first_row['short_commit_hash']}: {first_row['commit_message']})"
    )
    for row in rows:
        mean_seconds = float(row["mean_seconds"])
        factor, unit = _get_seconds_unit(mean_seconds)
        mean = _format_seconds(mean_seconds, factor, unit)
        stdev = _format_seconds(float(row["stdev_seconds"]), factor, unit)
        print(f"{row['benchmark_name']}: Mean +- std dev: {mean} +- {stdev}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m fnn.performance.report",
        description="Print pyperf-style summaries from benchmark CSV files.",
    )
    parser.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="Benchmark CSV file(s) to print. Defaults to the latest result.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = args.files or [_latest_result_file()]
    for index, path in enumerate(paths):
        if index:
            print()
        _print_file(path)


if __name__ == "__main__":
    main()
