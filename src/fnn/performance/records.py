import csv
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pyperf

from fnn.performance.scenarios import BenchmarkScenario

CSV_FIELDS = (
    "timestamp_utc",
    "commit_hash",
    "short_commit_hash",
    "commit_message",
    "working_tree_clean",
    "benchmark_name",
    "callable_name",
    "argument_description",
    "mean_seconds",
    "stdev_seconds",
    "median_seconds",
    "nrun",
    "nvalue",
    "unit",
    "python_version",
    "gil_disabled",
)
RESULTS_DIR = Path(__file__).with_name("results")
RESULTS_SUFFIX = "benchmarks.csv"


@dataclass(frozen=True)
class GitState:
    commit_hash: str
    short_commit_hash: str
    commit_message: str
    working_tree_clean: bool


def get_timestamp_utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def get_git_state() -> GitState:
    commit_hash = _git("rev-parse", "HEAD")
    return GitState(
        commit_hash=commit_hash,
        short_commit_hash=_git("rev-parse", "--short", "HEAD"),
        commit_message=_git("log", "-1", "--pretty=%s"),
        working_tree_clean=_git("status", "--porcelain") == "",
    )


def benchmark_to_row(
    benchmark: pyperf.Benchmark,
    scenario: BenchmarkScenario,
    git_state: GitState,
    timestamp_utc: str,
) -> dict[str, str]:
    return {
        "timestamp_utc": timestamp_utc,
        "commit_hash": git_state.commit_hash,
        "short_commit_hash": git_state.short_commit_hash,
        "commit_message": git_state.commit_message,
        "working_tree_clean": str(git_state.working_tree_clean),
        "benchmark_name": scenario.name,
        "callable_name": scenario.callable_name,
        "argument_description": scenario.argument_description,
        "mean_seconds": str(benchmark.mean()),
        "stdev_seconds": str(benchmark.stdev()),
        "median_seconds": str(benchmark.median()),
        "nrun": str(benchmark.get_nrun()),
        "nvalue": str(benchmark.get_nvalue()),
        "unit": benchmark.get_unit(),
        "python_version": sys.version,
        "gil_disabled": str(sysconfig.get_config_var("Py_GIL_DISABLED")),
    }


def write_benchmark_csv(rows: list[dict[str, str]], timestamp_utc: str) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    short_commit_hash = rows[0]["short_commit_hash"]
    path = RESULTS_DIR / f"{timestamp_utc}-{short_commit_hash}-{RESULTS_SUFFIX}"
    with path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _git(*args: str) -> str:
    return subprocess.check_output(("git", *args), text=True).strip()
