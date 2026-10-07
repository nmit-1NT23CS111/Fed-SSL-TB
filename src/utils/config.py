"""Load and validate the YAML config used by the encoder-only federated pipeline."""

import argparse
from types import SimpleNamespace

import yaml


REQUIRED_SECTIONS = ["data", "model", "ssl", "federated", "few_shot", "logging"]


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def _apply_override(config_dict: dict, key_path: str, value: str) -> None:
    keys = key_path.split(".")
    d = config_dict
    for key in keys[:-1]:
        if key not in d:
            raise KeyError(f"Config override key '{key}' not found in config.")
        d = d[key]

    raw = value
    if raw.lower() in ("true", "false"):
        coerced = raw.lower() == "true"
    else:
        try:
            coerced = int(raw)
        except ValueError:
            try:
                coerced = float(raw)
            except ValueError:
                coerced = raw
    d[keys[-1]] = coerced


def _ensure_aliases(config_dict: dict) -> None:
    if "few_shot" in config_dict and "finetuning" not in config_dict:
        config_dict["finetuning"] = config_dict["few_shot"]
    if "finetuning" in config_dict and "few_shot" not in config_dict:
        config_dict["few_shot"] = config_dict["finetuning"]
    if "loss" not in config_dict:
        config_dict["loss"] = {"lambda_mae": 0.70, "lambda_proto": 0.30}
    if "evaluation" not in config_dict:
        config_dict["evaluation"] = {"test_set": "montgomery", "metrics": ["auc", "accuracy", "sensitivity", "specificity", "f1", "balanced_accuracy"]}


def _validate(config_dict: dict) -> None:
    missing = [section for section in REQUIRED_SECTIONS if section not in config_dict]
    if missing:
        raise ValueError(f"Config is missing required sections: {missing}. Check the YAML file.")


def load_config(path: str = "configs/default.yaml") -> SimpleNamespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=path)
    known, unknown = parser.parse_known_args()

    with open(known.config, "r", encoding="utf-8") as handle:
        config_dict = yaml.safe_load(handle) or {}

    _ensure_aliases(config_dict)

    for arg in unknown:
        if not arg.startswith("--"):
            continue
        arg = arg[2:]
        if "=" in arg:
            key_path, value = arg.split("=", 1)
        else:
            key_path, value = arg, "true"
        try:
            _apply_override(config_dict, key_path, value)
        except KeyError as exc:
            print(f"[WARNING] CLI override ignored: {exc}")

    _validate(config_dict)
    namespace = _dict_to_namespace(config_dict)
    if hasattr(namespace, "few_shot") and not hasattr(namespace, "finetuning"):
        namespace.finetuning = namespace.few_shot
    if hasattr(namespace, "finetuning") and not hasattr(namespace, "few_shot"):
        namespace.few_shot = namespace.finetuning
    return namespace
