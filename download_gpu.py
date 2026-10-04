"""Install verified CUDA/cuDNN runtime DLLs locally, without replacing drivers."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import tarfile
import urllib.request
import zipfile
import zstandard

PACKAGES = [
 ('cudatoolkit-10.0.130-0.conda', '6fb67872b83a63f59a46509e8c28709b25a73363279ab085c0f049d055751844'),
 ('cudnn-7.6.5-cuda10.0_0.conda', '291587fe1bbbff0dc3154f3f5cf9e011b8264d124dedad5f257efa39726a4557')]
ROOT = Path(__file__).resolve().parent
TARGET = ROOT / '.runtime/gpu'

def install(package):
    name, expected = package
    url = 'https://repo.anaconda.com/pkgs/main/win-64/' + name
    cache = ROOT / '.runtime/downloads' / name
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists() or hashlib.sha256(cache.read_bytes()).hexdigest() != expected:
        print('Downloading', name, flush=True)
        urllib.request.urlretrieve(url, str(cache))
    actual = hashlib.sha256(cache.read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError('Package checksum mismatch: ' + name)
    installed = []
    with zipfile.ZipFile(str(cache)) as archive:
        payload = next(n for n in archive.namelist() if n.startswith('pkg-') and n.endswith('.tar.zst'))
        with archive.open(payload) as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                with tarfile.open(fileobj=stream, mode='r|') as tar:
                    for member in tar:
                        if member.isfile() and member.name.startswith('Library/bin/') and member.name.endswith('.dll'):
                            output = TARGET / member.name
                            output.resolve().relative_to(TARGET.resolve())
                            output.parent.mkdir(parents=True, exist_ok=True)
                            output.write_bytes(tar.extractfile(member).read())
                            installed.append(member.name)
    print('Installed', name, len(installed), 'DLLs', flush=True)
    return dict(package=name, url=url, sha256=actual, dlls=installed)

if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=2) as pool:
        evidence = list(pool.map(install, PACKAGES))
    (ROOT/'results').mkdir(exist_ok=True)
    (ROOT/'results/gpu_runtime_packages.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
