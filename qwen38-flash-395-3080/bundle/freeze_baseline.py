#!/usr/bin/env python3
"""Freeze the 395 + 3080 bundle: copy sources into the bundle, record hashes and
regenerate baseline.json plus SHA256SUMS. Idempotent: rerunning on an unchanged
tree reproduces the exact same baseline.json and SHA256SUMS.

    python bundle/freeze_baseline.py [--source <experiment workspace>]

Without --source the tree already inside bundle/ is the source of truth; with
--source the three cold-expert sources and the two reports are re-copied and
asserted to match the copies under bundle/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

BASE = Path(__file__).resolve().parent
ROOT = BASE.parent

SRC_DIR = BASE / 'src'
SOURCE_FILES = [
    ('strata_hipcold_lib_r435.cpp', SRC_DIR / 'strata_hipcold_lib_r435.cpp'),
    ('hipcold_bridge_r433.cpp', SRC_DIR / 'hipcold_bridge_r433.cpp'),
    ('hipcold_bridge_r433.hpp', SRC_DIR / 'hipcold_bridge_r433.hpp'),
    ('evidence/r434_report.md', ROOT / 'evidence' / 'r434_report.md'),
    ('evidence/r435_report.md', ROOT / 'evidence' / 'r435_report.md'),
]
FROZEN = [('bundle/runtime.env', BASE / 'runtime.env'),
          ('bundle/config/server.json', BASE / 'config' / 'server.json')]
METRICS = ROOT / 'evidence' / 'metrics.json'


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def reflash(source: Path) -> None:
    """Copy every source file over its bundle copy and assert the hashes match."""
    for name, target in SOURCE_FILES:
        upstream = source / Path(name).name
        if not upstream.is_file():
            raise SystemExit('missing source file: %s' % upstream)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(upstream, target)
        assert sha(upstream) == sha(target), name
    print('re-copied %d source artifacts from %s' % (len(SOURCE_FILES), source))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', help='experiment workspace holding the cold-expert sources')
    args = ap.parse_args()
    if args.source:
        reflash(Path(args.source))

    for _, path in SOURCE_FILES + FROZEN:
        if not path.is_file():
            raise SystemExit('missing bundle artifact: %s' % path)
    metrics = json.loads(METRICS.read_text(encoding='utf-8'))

    records = []
    for name, path in SOURCE_FILES + FROZEN:
        records.append({'path': name, 'sha256': sha(path), 'bytes': path.stat().st_size})
    baseline = {
        'schema': 'qwen38-flash-395-3080/baseline@1',
        'version': 'FLASH-395-01',
        'release': 'Qwen3.8-Flash-Next NVFP4 hot/cold expert split (RTX 3080 20GB + AI Max 395, single host)',
        'frozen_at': '2026-10-06',
        'status': 'accepted_local_baseline_snapshot',
        'architecture': {
            'host': 'single host (mx7)',
            'hot_gpu': {'device': 'RTX 3080 20GB', 'link': 'OCuLink', 'compute_capability': '8.6',
                        'cuda_arch': 'sm_86', 'work': 'hot experts and the dense main path'},
            'cold_compute': {'device': 'AI Max 395 / 8060S iGPU', 'arch': 'gfx1151',
                             'cpu_pool': 'Zen5 AVX-512',
                             'work': 'cold experts, either via HIP (STRATA_HIPDECODE=1, FRAC=128) or the CPU pool'},
            'engine': 'Strata NVFP4',
            'quantization': 'NVFP4',
            'kv': 'fp8',
            'max_context': 16384,
            'expert_cache': 4608,
            'kv_handoff_available': False,
        },
        'hip_cold_experts': {
            'enable_env': 'STRATA_HIPDECODE=1',
            'lib_env': 'STRATA_HIPCOLD_LIB',
            'frac': 128,
            'sync_mode': 'block',
            'slots': 384,
            'fallback': 'STRATA_HIPDECODE=0 runs cold experts on the AVX-512 CPU pool with no rebuild',
            'image_note': 'ROCm/HIP is not baked into the image; the gfx1151 library is compiled at container start and cached',
        },
        'performance': {
            'prefill_tok_s': metrics['prefill_tok_s'],
            'prefill_tok_s_approximate': metrics['prefill_tok_s_approximate'],
            'decode_tok_s': metrics['decode_tok_s'],
            'decode_tok_s_approximate': metrics['decode_tok_s_approximate'],
            'baseline_prefill_tok_s': metrics['baseline_prefill_tok_s'],
            'baseline_decode_tok_s': metrics['baseline_decode_tok_s'],
            'model': metrics['model'],
            'date': metrics['date'],
            'source': metrics['source'],
            'raw_logs_attached': metrics['raw_logs_attached'],
            'canonical_metrics': 'evidence/metrics.json',
        },
        'images': {
            'repository': 'ghcr.io/soulmate-halo/heterogeneous-gpu-pd-lab/qwen38-flash-395-3080',
            'tags': ['latest', 'sha-<commit>'],
            'platforms': ['linux/amd64'],
            'does_not_contain': ['125B Qwen3.8-Flash-Next NVFP4 checkpoint',
                                 'expert pack (experts.bin)', 'dense.gguf', 'PLE / embedding / MTP assets',
                                 'private credentials'],
            'runtime_preparation': 'Mount the converted assets under /models; the gfx1151 HIP cold-expert library is built on first start and cached in /cache',
        },
        'runtime_env': 'bundle/runtime.env',
        'mount_contract': ['STRATA_EXPERTS_PACK', 'STRATA_DENSE_GGUF', 'STRATA_PLE', 'STRATA_EMBED', 'STRATA_MTP'],
        'source': {
            'upstream': 'Strata plus local NVFP4/HIP-cold-expert changes (r433 bridge, r435 library)',
            'source_revision_sha256': hashlib.sha256(
                ''.join(r['sha256'] for r in records).encode('utf-8')).hexdigest(),
        },
        'artifacts': records,
    }
    text = json.dumps(baseline, ensure_ascii=False, indent=2) + '\n'
    for target in (BASE / 'baseline.json', ROOT / 'baseline.json'):
        target.write_text(text, encoding='utf-8')
    assert sha(BASE / 'baseline.json') == sha(ROOT / 'baseline.json')

    files = sorted((p for p in ROOT.rglob('*') if p.is_file()
                    and p.name not in ('SHA256SUMS',) and '__pycache__' not in p.parts
                    and not p.name.endswith('.pyc')), key=lambda p: p.relative_to(ROOT).as_posix())
    lines = [sha(p) + '  ' + p.relative_to(ROOT).as_posix() for p in files]
    (ROOT / 'SHA256SUMS').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('Frozen FLASH-395-01: %d source artifacts, %d checksums' % (len(records), len(lines)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
