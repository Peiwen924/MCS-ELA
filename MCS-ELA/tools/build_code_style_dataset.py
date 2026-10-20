#!/usr/bin/env python3
"""Build aligned Python/Java/C++ BioNER JSONL variants from NL JSONL."""
import argparse, json, re
from pathlib import Path

from bioner_common import iter_jsonl, write_jsonl

Q=lambda x:json.dumps(x,ensure_ascii=False)

def parse(row):
    m=row['messages']; p=m[0]['content']; a=p.rfind('Sentence: '); b=p.rfind('\nOutput:')
    if a<0 or b<a: raise ValueError('missing Sentence/Output markers')
    labels=re.findall(r'^- ([^:]+): (.+)$',p[:a],re.M)
    return p[a+10:b],labels,json.loads(m[-1]['content'])

def render(style,text,labels,entities):
    names=' | '.join(x[0] for x in labels)
    if style=='python':
        lm=', '.join(f'{Q(k)}: {Q(v)}' for k,v in labels)
        prompt=f'''from dataclasses import dataclass

@dataclass
class Entity:
    span: str
    type: str

class BioNER:
    label_set = {{{lm}}}

    def entity_recognition(self, text: str) -> list[str]:
        """Extract all biomedical entity mentions from text."""
        ...

    def type_classification(self, entity: str, text: str) -> str:
        """Predict the entity type: {names}."""
        ...

    def __call__(self, text: str) -> list[Entity]:
        return [Entity(span=e, type=self.type_classification(e, text)) for e in self.entity_recognition(text)]

bioner = BioNER()
text = {Q(text)}
output = bioner(text)

# Expected Python output:
/no_think'''
        vals=', '.join(f'Entity(span={Q(e["span"])}, type={Q(e["type"])})' for e in entities)
        answer=f'[{vals}]'
    elif style=='java':
        comments='\n'.join(f'    // {k}: {v}.' for k,v in labels)
        prompt=f'''record Entity(String span, String type) {{}}

class BioNER {{
{comments}
    List<String> entityRecognition(String text) {{ ... }}
    String typeClassification(String entity, String text) {{ /* {names} */ ... }}
    List<Entity> call(String text) {{
        List<Entity> output = new ArrayList<>();
        for (String entity : entityRecognition(text)) output.add(new Entity(entity, typeClassification(entity, text)));
        return output;
    }}
}}
BioNER bioner = new BioNER();
String text = {Q(text)};
List<Entity> output = bioner.call(text);

// Expected Java output:
/no_think'''
        vals=', '.join(f'new Entity({Q(e["span"])}, {Q(e["type"])})' for e in entities)
        answer=f'List.of({vals})' if entities else 'List.of()'
    else:
        comments='\n'.join(f'// {k}: {v}.' for k,v in labels)
        prompt=f'''struct Entity {{ std::string span; std::string type; }};
class BioNER {{
public:
    // Label Set:
{comments}
    std::vector<std::string> entity_recognition(const std::string& text) {{ ... }}
    std::string type_classification(const std::string& entity, const std::string& text) {{ /* {names} */ ... }}
    std::vector<Entity> operator()(const std::string& text) {{
        std::vector<Entity> output;
        for (const auto& entity : entity_recognition(text)) output.push_back(Entity{{entity, type_classification(entity, text)}});
        return output;
    }}
}};
BioNER bioner;
std::string text = {Q(text)};
std::vector<Entity> output = bioner(text);

// Expected C++ output:
/no_think'''
        vals=', '.join(f'Entity{{{Q(e["span"])}, {Q(e["type"])}}}' for e in entities)
        answer=f'std::vector<Entity>{{{vals}}}'
    return {'messages':[{'role':'user','content':prompt},{'role':'assistant','content':answer}]}

def convert(src, dst, style, entity_type=None):
    def records():
        for line_no, row in iter_jsonl(src):
            try:
                text, labels, entities = parse(row)
                if entity_type:
                    labels = [(label, description) for label, description in labels
                              if label.upper() == entity_type]
                    entities = [{"span": entity["span"], "type": entity_type} for entity in entities
                                if str(entity.get("type", "")).upper() == entity_type]
                yield render(style, text, labels, entities)
            except Exception as exc:
                raise ValueError(f"{src}:{line_no}: {exc}") from exc
    return write_jsonl(dst, records())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', default=['train', 'dev', 'test'])
    args = parser.parse_args()
    bc5cdr_types = {'bc5cdr_disease': 'DISEASE', 'bc5cdr_chemical': 'CHEMICAL'}
    entity_type = bc5cdr_types.get(args.data_dir.name)
    source_dir = args.data_dir.parent / 'bc5cdr' if entity_type else args.data_dir
    for split in args.splits:
        source = source_dir / f'nl_{split}.jsonl'
        for style in ('python', 'java', 'cpp'):
            destination = args.data_dir / f'{style}_{split}.jsonl'
            print(destination, convert(source, destination, style, entity_type))


if __name__ == '__main__':
    main()
