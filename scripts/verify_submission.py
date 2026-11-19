"""Verify submitted file bytes; standard library only, no training or GPU imports."""
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def verify(root=REPO):
    manifest = json.loads((root / 'SUBMISSION_MANIFEST.json').read_text(encoding='utf-8'))
    sources = json.loads((root / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))['files']
    errors = []
    for name, entry in manifest['files'].items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            errors.append(f'Unsafe manifest path: {name}')
        elif not path.is_file():
            errors.append(f'Missing: {name}')
        elif path.stat().st_size != entry['bytes'] or sha(path) != entry['sha256']:
            errors.append(f'Changed: {name}')
    for name, entry in sources.items():
        path = root / name
        if not path.is_file() or sha(path) != entry['sha256']:
            errors.append(f'Executed-source/result copy differs: {name}')
    if errors:
        raise ValueError('\n'.join(errors))
    return {'submission_files_checked': len(manifest['files']),
            'byte_identical_source_copies_checked': len(sources), 'status': 'PASS',
            'scope': 'Listed file bytes, not a full GPU experiment or an authenticity signature.'}


if __name__ == '__main__':
    try:
        print(json.dumps(verify(), indent=2))
    except (OSError, ValueError, KeyError) as error:
        sys.exit(str(error))
