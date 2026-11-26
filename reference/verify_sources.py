"""Verify the public reference bytes without accessing any private source."""
import hashlib
import json
from pathlib import Path


def verify():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = root / item['path']
        if path.resolve().parent != root:
            raise ValueError('Source path escapes reference directory')
        raw = path.read_bytes()
        if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
            raise ValueError('Reference source differs: '+item['path'])
    return {'status': 'PASS', 'source_files': len(manifest['files']),
            'scope': 'source identity only; not a benchmark replication claim'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
