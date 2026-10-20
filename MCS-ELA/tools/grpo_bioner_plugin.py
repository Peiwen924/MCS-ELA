import json
import re
from collections import Counter
from typing import List, Set, Tuple

from swift.rewards import ORM, orms

from bioner_common import extract_json_list, iter_code_entities, strip_thinking
from bioner_common import normalize_entity_span as normalize_span
from bioner_common import normalize_entity_type as normalize_type


Entity = Tuple[str, str]


def _parse_json_entities(text: str) -> Tuple[List[Entity], bool]:
    cleaned = strip_thinking(text)
    json_text = extract_json_list(cleaned)

    try:
        parsed = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return [], False

    if not isinstance(parsed, list):
        return [], False

    entities = []
    for item in parsed:
        if not isinstance(item, dict):
            return [], False
        span = normalize_span(item.get('span'))
        entity_type = item.get('type')
        if not span or entity_type is None:
            return [], False
        entities.append((span, normalize_type(entity_type)))

    return entities, cleaned == json_text


def parse_code_entities(text: str) -> Tuple[List[Entity], bool]:
    cleaned = strip_thinking(text)
    cleaned = cleaned.replace('<|im_end|>', '').strip()
    if not cleaned:
        return [], False

    empty_patterns = (
        r'^\[\s*\]$',
        r'^List\.of\s*\(\s*\)$',
        r'^std::vector\s*<\s*Entity\s*>\s*\{\s*\}$',
        r'^\{\s*\}$',
    )
    if any(re.match(pattern, cleaned, flags=re.DOTALL) for pattern in empty_patterns):
        return [], True

    entities = list(iter_code_entities(cleaned))
    return entities, bool(entities)


def parse_entities(text: str) -> Tuple[List[Entity], bool]:
    entities, valid = _parse_json_entities(text)
    if valid or entities:
        return entities, valid
    return parse_code_entities(text)


def precision_recall_f1(true_positive: int, false_positive: int, false_negative: int) -> Tuple[float, float, float]:
    if true_positive == false_positive == false_negative == 0:
        return 1.0, 1.0, 1.0

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def tokenize_span(span: str) -> List[str]:
    return re.findall(r'[A-Za-z0-9]+|[-+/()]', span.lower())


def span_similarity(pred_span: str, gold_span: str) -> float:
    pred_tokens = tokenize_span(pred_span)
    gold_tokens = tokenize_span(gold_span)
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def soft_boundary_f1(pred: Set[Entity], gold: Set[Entity]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0

    pred_items = list(pred)
    gold_items = list(gold)
    candidates = []
    for pred_index, (pred_span, pred_type) in enumerate(pred_items):
        for gold_index, (gold_span, gold_type) in enumerate(gold_items):
            if pred_type != gold_type:
                continue
            score = span_similarity(pred_span, gold_span)
            if score > 0:
                candidates.append((score, pred_index, gold_index))

    candidates.sort(reverse=True)
    used_pred = set()
    used_gold = set()
    matched_score = 0.0
    for score, pred_index, gold_index in candidates:
        if pred_index in used_pred or gold_index in used_gold:
            continue
        used_pred.add(pred_index)
        used_gold.add(gold_index)
        matched_score += score

    precision = matched_score / len(pred_items)
    recall = matched_score / len(gold_items)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


class BioNERFormatReward(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            _, strict_format = parse_entities(completion)
            if strict_format:
                rewards.append(1.0)
            elif extract_json_list(completion) != strip_thinking(completion):
                entities, valid = parse_entities(extract_json_list(completion))
                rewards.append(0.5 if valid or entities else 0.0)
            else:
                rewards.append(0.0)
        return rewards


class BioNERF1Reward(ORM):

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        rewards = []
        for completion, gold_text in zip(completions, solution):
            pred_entities, pred_valid = parse_entities(completion)
            gold_entities, gold_valid = parse_entities(gold_text)

            # Invalid output must not be interpreted as a correct empty prediction.
            if not pred_valid:
                rewards.append(-1.0)
                continue

            if not gold_valid:
                rewards.append(0.0)
                continue

            if not pred_entities and not gold_entities:
                rewards.append(1.0)
                continue

            if gold_entities and not pred_entities:
                rewards.append(-0.4)
                continue

            pred_set = set(pred_entities)
            gold_set = set(gold_entities)
            true_positive = len(pred_set & gold_set)
            false_positive = len(pred_set - gold_set)
            false_negative = len(gold_set - pred_set)
            precision, recall, exact_f1_score = precision_recall_f1(
                true_positive, false_positive, false_negative)
            boundary_score = soft_boundary_f1(pred_set, gold_set)

            if true_positive == 0 and boundary_score == 0:
                rewards.append(-0.2)
                continue

            reward = 0.55 * exact_f1_score
            reward += 0.20 * boundary_score
            reward += 0.125 * precision
            reward += 0.125 * recall
            reward -= 0.15 * false_negative / max(len(gold_set), 1)
            reward -= 0.15 * false_positive / max(len(pred_set), 1)

            rewards.append(max(-1.0, min(1.0, reward)))

        return rewards


orms['bioner_format'] = BioNERFormatReward
orms['bioner_f1'] = BioNERF1Reward
