#!/usr/bin/env python3
"""Train cooperative models across seeds and rank their epoch-150 test results."""

import argparse
import csv
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent.parent
DATASETS = {
    "vitro": ("in_vitro.json", 800),
    "silico_none": ("in_silico_noise_none.json", 6315),
    "silico_2": ("in_silico_noise_2.json", 6315),
    "silico_4": ("in_silico_noise_4.json", 6315),
}
FIELDS = [
    "seed", "epoch", "vitro_mm", "silico_none_mm",
    "silico_2_mm", "silico_4_mm", "score_mm", "checkpoint", "results_dir",
]


def positive_float(value):
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return result


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-seed", type=int, required=True)
    parser.add_argument("--max-trials", type=int, default=0,
                        help="number of consecutive seeds; 0 means no trial limit")
    parser.add_argument("--target-mm", type=positive_float,
                        help="stop when the selected test metric is at or below this value")
    parser.add_argument("--metric", choices=["vitro", "silico_none", "silico_2", "silico_4", "mean4"],
                        default="vitro")
    parser.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES value")
    parser.add_argument("--search-name", default="cooperative_cc68_seed_search")
    args = parser.parse_args()
    if args.start_seed < 0 or args.max_trials < 0:
        parser.error("--start-seed and --max-trials must be nonnegative")
    if args.max_trials == 0 and args.target_mm is None:
        parser.error("specify --max-trials or --target-mm to define a stopping condition")
    if not args.search_name or "/" in args.search_name or args.search_name in (".", ".."):
        parser.error("--search-name must be one directory name")
    return args


def score(row, metric):
    if metric == "mean4":
        return sum(float(row[f"{key}_mm"]) for key in DATASETS) / 4
    return float(row[f"{metric}_mm"])


def run(command, env):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def checkpoint_info(path):
    code = (
        "import json,sys,torch; "
        "s=torch.load(sys.argv[1],map_location='cpu',weights_only=False); "
        "print(json.dumps({'epoch':int(s['epoch'])}))"
    )
    result = subprocess.run(
        ["conda", "run", "-n", "geo_v2", "python", "-c", code, str(path)],
        cwd=ROOT, text=True, capture_output=True, check=True,
    )
    return json.loads(result.stdout.strip().splitlines()[-1])


def read_metrics(results_dir, checkpoint):
    values = {}
    for key, (name, expected_count) in DATASETS.items():
        path = results_dir / name
        if not path.is_file():
            return None
        try:
            if path.stat().st_mtime_ns < checkpoint.stat().st_mtime_ns:
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
            if Path(payload["checkpoint"]).resolve() != checkpoint.resolve():
                return None
            if int(payload["evaluated_samples"]) != expected_count:
                return None
            value = float(payload["method"]["mean_rms_tre_mm"])
            if not math.isfinite(value):
                return None
            values[f"{key}_mm"] = value
        except (KeyError, TypeError, ValueError, OSError):
            return None
    return values


def save_leaderboard(folder, rows, args):
    ordered = sorted(rows.values(), key=lambda row: (float(row["score_mm"]), int(row["seed"])))
    csv_path = folder / "leaderboard.csv"
    tmp_path = folder / "leaderboard.csv.tmp"
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(ordered)
    tmp_path.replace(csv_path)
    if ordered:
        best_path = folder / "best.json"
        tmp_path = folder / "best.json.tmp"
        tmp_path.write_text(json.dumps({
            "selection_metric": args.metric,
            "lower_is_better": True,
            "target_mm": args.target_mm,
            "best": ordered[0],
            "completed_seeds": sorted(rows),
        }, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(best_path)
    return ordered[0] if ordered else None


def main():
    args = arguments()
    folder = ROOT / "output" / args.search_name / args.metric
    folder.mkdir(parents=True, exist_ok=True)
    leaderboard = folder / "leaderboard.csv"
    rows = {}
    if leaderboard.exists():
        with leaderboard.open(newline="", encoding="utf-8") as handle:
            for old in csv.DictReader(handle):
                old["score_mm"] = score(old, args.metric)
                rows[int(old["seed"])] = old
    best = save_leaderboard(folder, rows, args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device
    env.pop("P2P_TEST_LIMIT", None)
    env.pop("P2P_TEST_INDICES", None)

    trials = range(args.max_trials) if args.max_trials else itertools.count()
    for offset in trials:
        if best and args.target_mm is not None and float(best["score_mm"]) <= args.target_mm:
            print(f"Target reached: {best['score_mm']:.6f} mm <= {args.target_mm:.6f} mm", flush=True)
            break
        seed = args.start_seed + offset
        train_name = f"rtor_a3_cooperative_cc68_seed{seed}"
        train_dir = ROOT / "output" / f"geotransformer.p2p_liver.{train_name}"
        final_epoch = train_dir / "snapshots" / "epoch-150.pth.tar"
        snapshot = train_dir / "snapshots" / "snapshot.pth.tar"
        test_name = f"{train_name}_epoch150"
        results_dir = ROOT / "output" / f"geotransformer.p2p_liver.{test_name}"
        print(f"\n=== Seed {seed} ===", flush=True)

        if not final_epoch.is_file():
            train_env = env.copy()
            train_env["P2P_RUN_NAME"] = train_name
            command = ["bash", "scripts/train_cooperative_seeded.sh", str(seed)]
            if snapshot.is_file():
                command.append("--resume")
            run(command, train_env)
        if not final_epoch.is_file():
            raise RuntimeError(f"Training did not finish 150 epochs: {train_dir}")

        values = read_metrics(results_dir, final_epoch)
        if values is None:
            test_env = env.copy()
            test_env["P2P_TRAIN_RUN_NAME"] = train_name
            test_env["P2P_TEST_RUN_NAME"] = test_name
            test_env["P2P_SNAPSHOT"] = str(final_epoch)
            run(["bash", "scripts/test_cooperative_seeded.sh", str(seed), "150"], test_env)
            values = read_metrics(results_dir, final_epoch)
            if values is None:
                raise RuntimeError(f"Four complete test outputs were not found: {results_dir}")

        info = checkpoint_info(final_epoch)
        if info["epoch"] != 150:
            raise RuntimeError(f"Expected epoch 150 in checkpoint, found {info['epoch']}: {final_epoch}")
        row = {
            "seed": seed,
            "epoch": info["epoch"],
            **values,
            "checkpoint": str(final_epoch),
            "results_dir": str(results_dir),
        }
        row["score_mm"] = score(row, args.metric)
        rows[seed] = row
        best = save_leaderboard(folder, rows, args)
        print(f"Seed {seed}: epoch {row['epoch']}, score {row['score_mm']:.6f} mm", flush=True)
        print(f"Current best: seed {best['seed']}, epoch {best['epoch']}, "
              f"{best['score_mm']:.6f} mm", flush=True)

    print(f"\nLeaderboard: {leaderboard}", flush=True)
    if best:
        print(f"Best checkpoint: {best['checkpoint']}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Stopped after an error: {error}", file=sys.stderr)
        sys.exit(1)
