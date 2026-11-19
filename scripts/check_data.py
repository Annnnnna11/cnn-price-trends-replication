"""Check the 54 expected image/label files against recorded input fingerprints."""
import argparse
import hashlib
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=REPO/'data/input_files.json')
    parser.add_argument('--sha256',action='store_true')
    args=parser.parse_args();manifest=json.loads(args.manifest.read_text(encoding='utf-8'))
    errors=[]
    for entry in manifest['files']:
        path=args.data_dir/entry['name']
        if not path.is_file():errors.append(f'Missing: {entry["name"]}');continue
        if path.stat().st_size!=entry['bytes']:errors.append(f'Size mismatch: {entry["name"]}');continue
        if args.sha256:
            value=hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
            if value.hexdigest()!=entry['sha256']:errors.append(f'Checksum mismatch: {entry["name"]}');continue
        print(f'PASS {entry["name"]}',flush=True)
    if errors:raise SystemExit('\n'.join(errors))
    print(f'DATA CHECK: {len(manifest["files"])} files; SHA256={args.sha256}',flush=True)


if __name__=='__main__':main()
