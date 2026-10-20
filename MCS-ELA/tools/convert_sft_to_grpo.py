import argparse
from pathlib import Path

from bioner_common import iter_jsonl, write_jsonl


def convert_record(record, line_no):
    messages = record.get('messages')
    if not isinstance(messages, list) or len(messages) < 2:
        raise ValueError(f'Line {line_no}: expected user and assistant messages')

    prompt_messages = []
    solution = None
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get('role') == 'assistant':
            solution = str(message.get('content', ''))
            break
        prompt_messages.append(message)

    if not prompt_messages:
        raise ValueError(f'Line {line_no}: no prompt messages found')
    if solution is None:
        raise ValueError(f'Line {line_no}: no assistant solution found')

    return {
        'messages': prompt_messages,
        'solution': solution,
        'labels': solution,
    }


def convert_file(input_path, output_path):
    records = (convert_record(record, line_no) for line_no, record in iter_jsonl(input_path))
    return write_jsonl(output_path, records)

def main():
    parser = argparse.ArgumentParser(description='Convert ms-swift SFT JSONL into GRPO prompt + solution JSONL.')
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    count = convert_file(Path(args.input), Path(args.output))
    print(f'Converted {count} records: {args.input} -> {args.output}')


if __name__ == '__main__':
    main()
