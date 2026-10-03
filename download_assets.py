"""Download upstream checkpoint or genuine LJSpeech pairs without the full archive."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

BASE = 'https://data.keithito.com/data/speech/'


def checkpoint():
    root = Path('checkpoints')
    root.mkdir(exist_ok=True)
    target = root / 'tacotron-20180906'
    if (target / 'model.ckpt.index').exists() and (target / 'model.ckpt.data-00000-of-00001').exists():
        print('Checkpoint already present:', target)
        return
    with urllib.request.urlopen(BASE + 'tacotron-20180906.tar.gz', timeout=60) as response:
        with tarfile.open(fileobj=response, mode='r|gz') as archive:
            for member in archive:
                path = root / member.name
                try:
                    path.resolve().relative_to(root.resolve())
                except ValueError:
                    raise ValueError('Unsafe archive path')
                if member.isfile():
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(archive.extractfile(member).read())
    print('Downloaded upstream checkpoint:', target)


def subset(count):
    root = Path('LJSpeech-1.1')
    if (root / 'metadata.csv').exists():
        raise RuntimeError('Dataset exists; move it aside explicitly before downloading a new subset.')
    (root / 'wavs').mkdir(parents=True, exist_ok=True)
    ids, rows = [], None
    with urllib.request.urlopen(BASE + 'LJSpeech-1.1.tar.bz2', timeout=60) as response:
        with tarfile.open(fileobj=response, mode='r|bz2') as archive:
            for member in archive:
                if member.name.endswith('metadata.csv'):
                    rows = archive.extractfile(member).read().decode('utf-8').splitlines()
                elif member.isfile() and member.name.endswith('.wav'):
                    name = Path(member.name).name
                    (root / 'wavs' / name).write_bytes(archive.extractfile(member).read())
                    ids.append(name[:-4])
                    print('Downloaded', name, flush=True)
                if rows is not None and len(ids) >= count:
                    break
    if rows is None or len(ids) < count:
        raise RuntimeError('Archive did not contain the requested metadata/audio pairs.')
    selected = [row for row in rows if row.split('|')[0] in ids]
    if len(selected) != len(ids):
        raise RuntimeError('Some downloaded WAVs do not have transcripts.')
    (root / 'metadata.csv').write_text('\n'.join(selected) + '\n', encoding='utf-8')
    manifest = dict(source=BASE+'LJSpeech-1.1.tar.bz2', selection='first WAV members in official archive',
                    samples=len(ids), ids=ids,
                    sha256={name: hashlib.sha256((root/'wavs'/(name+'.wav')).read_bytes()).hexdigest() for name in ids})
    Path('results').mkdir(exist_ok=True)
    Path('results/dataset_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Saved %d genuine LJSpeech pairs' % len(selected))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', action='store_true')
    parser.add_argument('--subset', type=int)
    args = parser.parse_args()
    if not args.checkpoint and args.subset is None:
        parser.error('Specify --checkpoint and/or --subset N.')
    if args.subset is not None and args.subset < 1:
        parser.error('--subset must be positive.')
    if args.checkpoint:
        checkpoint()
    if args.subset:
        subset(args.subset)


if __name__ == '__main__':
    main()
