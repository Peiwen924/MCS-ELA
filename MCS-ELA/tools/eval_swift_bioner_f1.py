"""Evaluate predictions, vote across three styles, or select a saved checkpoint."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from bioner_common import (STYLES, extract_json_list, iter_code_entities, normalize_entity_span,
                          normalize_entity_type, read_jsonl, strip_thinking,
                          update_best_symlink, write_checkpoint_summary)


def entity_id(span, entity_type):
    return f"{normalize_entity_span(span)}#{normalize_entity_type(entity_type)}"


def parse_bioner_output(text):
    if not text:
        return set()

    text = strip_thinking(text)
    json_text = extract_json_list(text)
    if not json_text or json_text.upper() == "NULL" or json_text in {"[]", "{}"}:
        return set()

    entities = set()
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            span = item.get("span") or item.get("entity") or item.get("name") or item.get("text")
            entity_type = item.get("type") or item.get("label")
            span = normalize_entity_span(span)
            if span:
                entities.add(entity_id(span, entity_type))
        if entities:
            return entities

    for span, entity_type in iter_code_entities(text):
        entities.add(entity_id(span, entity_type))

    dict_matches = re.findall(
        r'["\'](?:span|entity|name|text)["\']\s*:\s*["\']([^"\']+)["\']\s*,\s*'
        r'["\'](?:type|label)["\']\s*:\s*["\']([^"\']+)["\']',
        text,
    )
    for span, entity_type in dict_matches:
        span = normalize_entity_span(span)
        if span:
            entities.add(entity_id(span, entity_type))

    return entities


def compute_micro_prf(records, pred_key="response", label_key="labels"):
    true_positive = 0
    pred_total = 0
    label_total = 0
    detailed = []

    for index, record in enumerate(records):
        pred_text = record.get(pred_key, "")
        label_text = record.get(label_key, "")
        pred_set = parse_bioner_output(pred_text)
        label_set = parse_bioner_output(label_text)

        true_positive += len(pred_set & label_set)
        pred_total += len(pred_set)
        label_total += len(label_set)

        detailed.append(
            {
                "index": index,
                "prediction": pred_text,
                "label": label_text,
                "parsed_prediction": sorted(pred_set),
                "parsed_label": sorted(label_set),
                "true_positive": sorted(pred_set & label_set),
                "false_positive": sorted(pred_set - label_set),
                "false_negative": sorted(label_set - pred_set),
            }
        )

    metrics = micro_metrics(true_positive, pred_total, label_total, len(records))
    return metrics, detailed

def micro_metrics(true_positive, pred_total, label_total, samples):
    precision = true_positive / pred_total if pred_total else 0.0
    recall = true_positive / label_total if label_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "predicted": pred_total,
        "gold": label_total,
        "samples": samples,
    }


def compute_vote_metrics(paths, pred_key="response", label_key="labels", save_details=False):
    """Vote directly on parsed entity sets, without serializing and reparsing them."""
    rows = {style: read_jsonl(paths[style]) for style in STYLES}
    lengths = {style: len(records) for style, records in rows.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Prediction files differ in length: {lengths}")
    samples = next(iter(lengths.values()))
    totals = {threshold: [0, 0, 0] for threshold in (1, 2, 3)}
    details = []
    for index in range(samples):
        predictions = {
            style: parse_bioner_output(rows[style][index].get(pred_key, "")) for style in STYLES
        }
        labels = {
            style: parse_bioner_output(rows[style][index].get(label_key, "")) for style in STYLES
        }
        if len({frozenset(value) for value in labels.values()}) != 1:
            raise ValueError(f"Gold labels are not aligned at zero-based row {index}: {labels}")
        gold = next(iter(labels.values()))
        counts = Counter(entity for entities in predictions.values() for entity in entities)
        for threshold, total in totals.items():
            selected = {entity for entity, count in counts.items() if count >= threshold}
            total[0] += len(selected & gold)
            total[1] += len(selected)
            total[2] += len(gold)
        if save_details:
            details.append({
                "index": index,
                "predictions": {style: sorted(value) for style, value in predictions.items()},
                "votes": dict(sorted(counts.items())),
                "labels": sorted(gold),
            })
    result = {
        "inputs": paths, "samples": samples,
        "thresholds": {str(threshold): micro_metrics(*total, samples) for threshold, total in totals.items()},
    }
    if save_details:
        result["records"] = details
    return result


def select_checkpoint(args):
    output_dir = Path(args.output_dir)
    summary_path = Path(args.summary) if args.summary else output_dir / "dev_checkpoint_metrics.json"
    rows = []
    for checkpoint in sorted(output_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])):
        metrics_path = checkpoint / args.metrics_name
        if metrics_path.exists():
            with metrics_path.open(encoding="utf-8") as handle:
                f1 = float(json.load(handle)["metrics"]["f1"])
            rows.append({"checkpoint": str(checkpoint), "metrics_path": str(metrics_path), "f1": f1})
    if not rows:
        raise SystemExit(f"No checkpoint metrics found under {output_dir}/checkpoint-*/{args.metrics_name}")
    best = write_checkpoint_summary(rows, summary_path, "strict dev micro-F1")
    update_best_symlink(output_dir, best["checkpoint"], args.symlink_name)
    print(f"Best checkpoint by strict dev micro-F1: {best['checkpoint']}")
    print(f"Best dev micro-F1: {best['f1']:.6f}")
    print(f"Summary: {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="BioNER evaluation, three-style voting, and checkpoint selection.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="Evaluate a swift infer JSONL file.")
    source.add_argument("--python", help="Vote on Python/Java/C++ prediction files.")
    source.add_argument("--output_dir", help="Select the best checkpoint from saved dev metrics.")
    parser.add_argument("--java")
    parser.add_argument("--cpp")
    parser.add_argument("--output", help="JSON summary output; required for voting.")
    parser.add_argument("--pred_key", default="response")
    parser.add_argument("--label_key", default="labels")
    parser.add_argument("--save_details", action="store_true")
    parser.add_argument("--metrics_name", default="dev_metrics.json")
    parser.add_argument("--symlink_name", default="best_dev_f1")
    parser.add_argument("--summary", help="Checkpoint summary path.")
    args = parser.parse_args()
    if args.python and not (args.java and args.cpp and args.output):
        parser.error("Voting requires --python, --java, --cpp and --output.")
    if not args.python and (args.java or args.cpp):
        parser.error("--java and --cpp are only used with --python.")
    return args


def main():
    args = parse_args()
    if args.output_dir:
        select_checkpoint(args)
        return
    if args.python:
        result = compute_vote_metrics(
            {style: getattr(args, style) for style in STYLES},
            args.pred_key, args.label_key, args.save_details,
        )
        for threshold, metrics in result["thresholds"].items():
            print(f"votes>={threshold}: P/R/F1="
                  f"{metrics['precision']:.6f}/{metrics['recall']:.6f}/{metrics['f1']:.6f}")
    else:
        metrics, detailed = compute_micro_prf(read_jsonl(args.input), args.pred_key, args.label_key)
        result = {"metrics": metrics}
        if args.save_details:
            result["records"] = detailed
        print(f"BioNER micro P/R/F1 = "
              f"{metrics['precision']:.6f}/{metrics['recall']:.6f}/{metrics['f1']:.6f}")
        print(f"Counts: TP={metrics['true_positive']}, Pred={metrics['predicted']}, "
              f"Gold={metrics['gold']}, Samples={metrics['samples']}")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved evaluation summary to {output}")


if __name__ == "__main__":
    main()
