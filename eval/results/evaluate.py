#!/usr/bin/env python3
"""Score AoH LongBench predictions and reproduce the paper's categories."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

try:
    from .metrics import (
        classification_score,
        code_sim_score,
        count_score,
        qa_f1_score,
        qa_f1_zh_score,
        retrieval_score,
        retrieval_zh_score,
        rouge_score,
        rouge_zh_score,
    )
except ImportError:
    from metrics import (
        classification_score,
        code_sim_score,
        count_score,
        qa_f1_score,
        qa_f1_zh_score,
        retrieval_score,
        retrieval_zh_score,
        rouge_score,
        rouge_zh_score,
    )

TASK_METRICS = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_count": count_score,
    "passage_retrieval_en": retrieval_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}

CATEGORIES = {
    "SQA": ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh"],
    "MQA": ["hotpotqa", "2wikimqa", "musique", "dureader"],
    "Summ": ["gov_report", "qmsum", "multi_news", "vcsum"],
    "Fewshot": ["trec", "triviaqa", "samsum", "lsht"],
    "Synthetic": ["passage_count", "passage_retrieval_en", "passage_retrieval_zh"],
    "Code": ["lcc", "repobench-p"],
}


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def score_task(path: Path, task: str) -> float:
    metric = TASK_METRICS[task]
    scores = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            sample = json.loads(line)
            prediction = sample["pred"]
            if task in {"trec", "triviaqa", "samsum", "lsht"}:
                prediction = prediction.lstrip("\n").split("\n")[0]
            best = max(
                metric(
                    prediction,
                    answer,
                    all_classes=sample.get("all_classes"),
                )
                for answer in sample["answers"]
            )
            scores.append(best)
    return round(100 * mean(scores), 2)


def evaluate_directory(prediction_dir: Path) -> dict:
    task_scores = {}
    missing = []
    for task in TASK_METRICS:
        path = prediction_dir / f"{task}.jsonl"
        if path.is_file():
            task_scores[task] = score_task(path, task)
        else:
            missing.append(task)

    category_scores = {}
    for category, tasks in CATEGORIES.items():
        available = [task_scores[task] for task in tasks if task in task_scores]
        if available:
            category_scores[category] = round(mean(available), 2)

    overall = round(mean(list(task_scores.values())), 2) if not missing else None
    return {
        "prediction_dir": str(prediction_dir),
        "task_scores": task_scores,
        "category_scores": category_scores,
        "average": overall,
        "missing_tasks": missing,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = evaluate_directory(args.prediction_dir)
    output = args.output or args.prediction_dir / "scores.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    row = {**result["category_scores"], "Avg": result["average"]}
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=[*CATEGORIES, "Avg"])
        writer.writeheader()
        writer.writerow(row)

    print(json.dumps(row, indent=2))
    if result["missing_tasks"]:
        print("Missing tasks: " + ", ".join(result["missing_tasks"]))


if __name__ == "__main__":
    main()
