from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import yaml

from ..config import load_config, asdict_config
from .experiment import run_experiment
from ..utils.io import save_yaml


def _flatten(prefix: str, d: dict) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten(key, v))
        else:
            out[key] = v
    return out


def _merge_overrides(base: Dict[str, object], extra: Dict[str, object]) -> Dict[str, object]:
    out = dict(base)
    out.update(extra)
    return out


def _load_yaml_dict(p: str | Path) -> dict:
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="psr", description="Poincaré Section Reservoirs CLI")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Run an experiment")
    p_train.add_argument("--config", type=str, default=None, help="Base config YAML")
    p_train.add_argument("--dataset", type=str, default=None, help="Dataset YAML to merge")
    p_train.add_argument("--model", type=str, default=None, help="Model YAML to merge")
    p_train.add_argument("--train", type=str, default=None, help="Train YAML to merge")
    p_train.add_argument("--eval", type=str, default=None, help="Eval YAML to merge")
    p_train.add_argument("--set", nargs="*", help="Dotted overrides like model.bins=64 train.seed=1")

    p_plot = sub.add_parser("plot", help="(Placeholder) Show where plots were saved")
    p_plot.add_argument("--run", type=str, required=True, help="Run directory")

    args = parser.parse_args(argv)

    if args.cmd == "train":
        overrides: Dict[str, object] = {}
        if args.dataset:
            overrides = _merge_overrides(overrides, _flatten("dataset", _load_yaml_dict(args.dataset)))
        if args.model:
            overrides = _merge_overrides(overrides, _flatten("model", _load_yaml_dict(args.model)))
        if args.train:
            overrides = _merge_overrides(overrides, _flatten("train", _load_yaml_dict(args.train)))
        if args.eval:
            overrides = _merge_overrides(overrides, _flatten("eval", _load_yaml_dict(args.eval)))

        # parse --set key=value pairs
        if args.set:
            for kv in args.set:
                k, v = kv.split("=", 1)
                # attempt type-cast
                if v.isdigit():
                    v_cast: object = int(v)
                else:
                    try:
                        v_cast = float(v)
                    except ValueError:
                        if v.lower() in {"true", "false"}:
                            v_cast = v.lower() == "true"
                        else:
                            v_cast = v
                overrides[k] = v_cast

        cfg = load_config(args.config, overrides=overrides if overrides else None)
        result = run_experiment(cfg)
        print("Run dir:", result.run_dir)
        print("NRMSE by horizon:", result.metrics)
        # also save summary
        save_yaml(Path(result.run_dir) / "summary.yaml", {"metrics": result.metrics})

    elif args.cmd == "plot":
        run_dir = Path(args.run)
        if not run_dir.exists():
            raise SystemExit(f"Run not found: {run_dir}")
        print(f"Figures are in: {run_dir}")
        print("Files:")
        for p in sorted(run_dir.glob("*.png")):
            print("  -", p.name)


if __name__ == "__main__":
    main()
