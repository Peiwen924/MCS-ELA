"""Shared BioNER text parsing and JSONL helpers (standard library only)."""
import json
import os
import re
from pathlib import Path


STYLES = ("python", "java", "cpp")


def normalize_entity_span(span):
    return re.sub(r"\s+", " ", str(span or "").strip())


def normalize_entity_type(entity_type):
    label = re.sub(r"[^0-9A-Za-z_]+", "_", str(entity_type or "ENTITY").strip())
    label = re.sub(r"_+", "_", label).strip("_").upper() or "ENTITY"
    return f"ENTITY_{label}" if label[0].isdigit() else label


def strip_thinking(text):
    text = str(text or "")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    return text.replace("```", "").strip()


def extract_json_list(text):
    text = strip_thinking(text)
    start = text.find("[")
    end = text.rfind("]")
    if start >= 0 and end >= start:
        return text[start:end + 1].strip()
    return text


def _unescape_quoted(value):
    """Remove matching quotes and decode common code-string escapes."""
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return re.sub(r'\\([\\"\'])', r'\1', value)


def iter_code_entities(text):
    """Yield Python/Java/C++ entities, preserving duplicates for reward parsing."""
    value = r"""(?P<{name}>"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')"""
    patterns = (
        r'Entity\s*\(\s*span\s*=\s*' + value.format(name="span")
        + r'\s*,\s*type\s*=\s*' + value.format(name="entity_type") + r'\s*\)',
        r'(?:new\s+)?Entity\s*\(\s*' + value.format(name="span")
        + r'\s*,\s*' + value.format(name="entity_type") + r'\s*\)',
        r'(?:Entity\s*)?\{\s*' + value.format(name="span")
        + r'\s*,\s*' + value.format(name="entity_type") + r'\s*\}',
    )
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.DOTALL):
            span = normalize_entity_span(_unescape_quoted(match.group("span")))
            if span:
                yield span, normalize_entity_type(_unescape_quoted(match.group("entity_type")))



def iter_jsonl(path):
    """Yield (physical line number, record), skipping blank lines and a UTF-8 BOM."""
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield line_no, json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_no}") from exc


def read_jsonl(path):
    return [record for _, record in iter_jsonl(path)]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for count, record in enumerate(records, start=1):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return count


def public_records(records):
    return [{"messages": record["messages"]} for record in records]


def write_checkpoint_summary(rows, path, selection_target, **metadata):
    best = max(rows, key=lambda row: row["f1"])
    summary = {
        "selection_target": selection_target,
        **metadata,
        "best_checkpoint": best["checkpoint"],
        "best_f1": best["f1"],
        "checkpoints": rows,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return best



def update_best_symlink(output_dir, checkpoint, name="best_dev_f1"):
    output_dir = Path(output_dir)
    temporary = output_dir / f".{name}.tmp"
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    os.symlink(Path(checkpoint).name, temporary)
    os.replace(temporary, output_dir / name)

