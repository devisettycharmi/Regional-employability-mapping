from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def compute_matching_accuracy(top_matches: pd.DataFrame, target_column: str = "target_role") -> dict[str, float]:
    correct_top1 = 0
    correct_top3 = 0
    total = len(top_matches)
    for _, row in top_matches.iterrows():
        roles = row.get("top3_roles", [])
        target = str(row.get(target_column, "")).strip().lower()
        if target and roles:
            if roles[0].lower() == target:
                correct_top1 += 1
            if any(role.lower() == target for role in roles):
                correct_top3 += 1
    return {"top1_accuracy": correct_top1 / total if total else 0.0, "top3_accuracy": correct_top3 / total if total else 0.0}


def export_manual_review_csv(students: pd.DataFrame, path: str | Path) -> None:
    sample = students.sample(n=min(50, len(students)), random_state=42)
    sample.to_csv(path, index=False)


def build_evaluation_report(metrics: dict) -> str:
    lines = ["# Evaluation report", "\n"]
    for key, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value}")
    return "\n".join(lines)


if __name__ == "__main__":
    print("Evaluation helpers ready.")
