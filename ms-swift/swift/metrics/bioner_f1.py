# Copyright (c) ModelScope Contributors. All rights reserved.
import json
import re
from typing import Dict, Iterable, List, Set

from transformers import EvalPrediction

from swift.utils import Serializer
from .base import EvalMetrics


def normalize_entity_span(span) -> str:
    return re.sub(r'\s+', ' ', str(span or '').strip())


def normalize_entity_type(entity_type) -> str:
    label = str(entity_type or 'ENTITY').strip()
    label = re.sub(r'[^0-9A-Za-z_]+', '_', label)
    label = re.sub(r'_+', '_', label).strip('_').upper()
    if not label:
        label = 'ENTITY'
    if label[0].isdigit():
        label = 'ENTITY_' + label
    return label


def entity_id(span, entity_type) -> str:
    return f'{normalize_entity_span(span)}#{normalize_entity_type(entity_type)}'


def strip_thinking(text: str) -> str:
    text = str(text or '')
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'```(?:json)?\s*', '', text, flags=re.IGNORECASE)
    text = text.replace('```', '')
    return text.strip()


def extract_json_list(text: str) -> str:
    text = strip_thinking(text)
    start = text.find('[')
    end = text.rfind(']')
    if start >= 0 and end >= start:
        return text[start:end + 1].strip()
    return text


def parse_bioner_output(text: str) -> Set[str]:
    if not text:
        return set()

    text = extract_json_list(text)
    if not text or text.upper() == 'NULL' or text in {'[]', '{}'}:
        return set()

    entities = set()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            span = item.get('span') or item.get('entity') or item.get('name') or item.get('text')
            entity_type = item.get('type') or item.get('label')
            span = normalize_entity_span(span)
            if span:
                entities.add(entity_id(span, entity_type))
        if entities:
            return entities

    # Tolerate common JSON-like outputs where the model used quotes correctly
    # but wrapped the result in extra text.
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


def micro_prf(pred_sets: Iterable[Set[str]], label_sets: Iterable[Set[str]]) -> Dict[str, float]:
    true_positive = 0
    pred_total = 0
    label_total = 0

    for pred, label in zip(pred_sets, label_sets):
        true_positive += len(pred & label)
        pred_total += len(pred)
        label_total += len(label)

    precision = true_positive / pred_total if pred_total else 0.0
    recall = true_positive / label_total if label_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        'precision': round(precision * 100, 6),
        'recall': round(recall * 100, 6),
        'f1': round(f1 * 100, 6),
    }


class BionerF1Metrics(EvalMetrics):

    def compute_metrics(self, eval_prediction: EvalPrediction) -> Dict[str, float]:
        preds, labels = eval_prediction.predictions, eval_prediction.label_ids
        pred_sets: List[Set[str]] = []
        label_sets: List[Set[str]] = []

        for i in range(preds.shape[0]):
            pred_text = Serializer.from_tensor(preds[i])
            label_text = Serializer.from_tensor(labels[i])
            pred_sets.append(parse_bioner_output(pred_text))
            label_sets.append(parse_bioner_output(label_text))

        return micro_prf(pred_sets, label_sets)
