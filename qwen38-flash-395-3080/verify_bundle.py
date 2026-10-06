#!/usr/bin/env python3
"""Verify the single-host 395 + RTX 3080 bundle without CUDA, HIP or model weights."""
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIRED = ['README.md', 'README_ZH.md', 'Dockerfile', 'docker-compose.yml', 'entrypoint.sh',
            'apply-bundle.sh', 'baseline.json', 'bundle/baseline.json', 'bundle/runtime.env',
            'bundle/config/server.json', 'bundle/src/strata_hipcold_lib_r435.cpp',
            'bundle/src/hipcold_bridge_r433.cpp', 'bundle/src/hipcold_bridge_r433.hpp',
            'bundle/freeze_baseline.py', 'evidence/metrics.json', 'evidence/r434_report.md',
            'evidence/r435_report.md', 'scripts/container-start.sh', 'scripts/doctor.py',
            'scripts/health.sh', 'scripts/help.sh', 'scripts/smoke.sh']
TEXT_SUFFIXES = {'.md', '.txt', '.json', '.csv', '.py', '.sh', '.yml', '.yaml', '.env', '.cpp', '.hpp', '.h', '.cu'}
WIN_USERS = 'C:' + chr(92) + 'Users' + chr(92)
FORBIDDEN = (WIN_USERS, '/home/ysy', 'gho_', 'sk-proj-', 'BEGIN OPENSSH PRIVATE KEY')


def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_sums(errors):
    sums = ROOT / 'SHA256SUMS'
    if not sums.is_file():
        errors.append('missing SHA256SUMS')
        return
    for no, line in enumerate(sums.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip() or line.startswith('#'):
            continue
        parts = line.split('  ', 1)
        if len(parts) != 2:
            errors.append(f'SHA256SUMS:{no}: invalid format')
            continue
        expected, rel = parts
        rel = rel.strip().replace(chr(92), '/')
        p = (ROOT / rel).resolve()
        if ROOT not in p.parents or not p.is_file():
            errors.append(f'SHA256SUMS:{no}: missing or escaping {rel}')
            continue
        if digest(p) != expected:
            errors.append(f'SHA256SUMS:{no}: hash mismatch {rel}')


def verify_archives(errors):
    """Same tgz safety contract as the Spark kit; this bundle ships no config package."""
    packages = ROOT / 'bundle' / 'packages'
    if not packages.is_dir():
        return
    for p in sorted(packages.glob('*.tgz')):
        try:
            with tarfile.open(p, 'r:gz') as tar:
                for m in tar.getmembers():
                    if m.name.startswith('/') or '..' in Path(m.name).parts:
                        errors.append(f'unsafe archive member {p.name}:{m.name}')
                if not any(m.isfile() for m in tar.getmembers()):
                    errors.append(f'empty archive {p.name}')
        except (OSError, tarfile.TarError) as exc:
            errors.append(f'invalid archive {p.name}: {exc}')


def verify_metrics(errors):
    try:
        m = json.loads((ROOT / 'evidence/metrics.json').read_text(encoding='utf-8'))
        if m['prefill_tok_s'] != 800 or m['prefill_tok_s_approximate'] is not True:
            errors.append('prefill frozen value changed')
        if m['decode_tok_s'] != 40 or m['decode_tok_s_approximate'] is not True:
            errors.append('decode frozen value changed')
        if m['baseline_prefill_tok_s'] != 273.04:
            errors.append('baseline prefill value changed')
        if m['baseline_decode_tok_s'] != 41.33:
            errors.append('baseline decode value changed')
        if m['model'] != 'qwen3.8-flash-next' or m['date'] != '2026-10-06':
            errors.append('model or date changed')
        if m['raw_logs_attached'] is not False:
            errors.append('raw log disclosure missing')
    except Exception as exc:
        errors.append(f'metrics invalid: {exc}')


def scan_public_text(errors):
    roots = [ROOT / 'README.md', ROOT / 'README_ZH.md', ROOT / 'Dockerfile', ROOT / 'docker-compose.yml',
             ROOT / 'entrypoint.sh', ROOT / 'apply-bundle.sh', ROOT / 'bundle', ROOT / 'evidence',
             ROOT / 'scripts', ROOT / 'baseline.json']
    for base in roots:
        paths = [base] if base.is_file() else list(base.rglob('*'))
        for p in paths:
            if not p.is_file() or p.name == 'SHA256SUMS' or p.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = p.read_text(encoding='utf-8', errors='ignore')
            for marker in FORBIDDEN:
                if marker in text:
                    errors.append(f'private marker {marker!r} in {p.relative_to(ROOT)}')


def main():
    errors = []
    for rel in REQUIRED:
        if not (ROOT / rel).is_file():
            errors.append(f'missing {rel}')
    verify_sums(errors)
    verify_archives(errors)
    verify_metrics(errors)
    scan_public_text(errors)
    if errors:
        print('FAIL')
        print('\n'.join('- ' + e for e in errors))
        return 1
    print('PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
