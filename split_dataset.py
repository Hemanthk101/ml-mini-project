"""Deterministic transcript-group split for future training and held-out evaluation."""
import argparse, hashlib, json, random
from pathlib import Path

def split_rows(rows, seed=2026):
    groups={}
    for row in rows:
        key=' '.join(row.split('|',3)[3].casefold().split())
        groups.setdefault(key,[]).append(row)
    keys=sorted(groups); random.Random(seed).shuffle(keys)
    if len(keys)<3: raise ValueError('At least three distinct transcript groups are needed.')
    count=max(1,len(keys)//10)
    partitions={'validation':keys[:count],'test':keys[count:2*count],'train':keys[2*count:]}
    return {name:[row for key in selected for row in groups[key]] for name,selected in partitions.items()}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',default='training/train.txt');p.add_argument('--seed',type=int,default=2026);a=p.parse_args()
    source=Path(a.input); rows=[x for x in source.read_text(encoding='utf-8').splitlines() if x.strip()]
    split=split_rows(rows,a.seed); manifest={'seed':a.seed,'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'grouping':'Casefolded normalized transcript; duplicate transcripts stay together','partitions':{},'scope':'Held out for a NEW run only. Upstream LJSpeech-pretrained checkpoints may already have seen these examples.'}
    for name,selected in split.items():
        destination=source.parent/(name+'_split.txt');destination.write_text('\n'.join(selected)+'\n',encoding='utf-8')
        manifest['partitions'][name]={'path':str(destination),'count':len(selected),'spectrograms':[x.split('|')[0] for x in selected]}
    target=source.parent/'split_manifest.json';target.write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
