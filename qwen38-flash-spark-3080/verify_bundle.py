#!/usr/bin/env python3
"""Verify the public Qwen3.8 Flash/Spark bundle without CUDA or model weights."""
from __future__ import annotations
import hashlib, json, tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parent
REQUIRED = ['README.md','README_ZH.md','NOTICE','Dockerfile','.dockerignore','docker-compose.yml','entrypoint.sh','apply-bundle.sh','run-dual-host.example.sh','baseline.json','bundle/baseline.json','bundle/runtime.env','bundle/config/engine-3080.serve.json','bundle/config/worker-spark.json','bundle/config/router.json','evidence/v2-metrics.json','evidence/performance-v2.json','evidence/README.md','scripts/container-help.sh','scripts/container-health.sh','scripts/container-smoke.sh','scripts/container-start.sh','scripts/container-build.sh','scripts/deploy.sh','scripts/doctor.py']
TEXT_SUFFIXES = {'.md','.txt','.json','.csv','.py','.sh','.yml','.yaml','.toml','.env'}
FORBIDDEN = ('C:\\Users\\','/home/ysy','gho_','sk-proj-','BEGIN OPENSSH PRIVATE KEY')
def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def verify_sums(errors):
    sums=ROOT/'SHA256SUMS'
    if not sums.is_file(): errors.append('missing SHA256SUMS'); return
    seen=set()
    for no,line in enumerate(sums.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip() or line.startswith('#'): continue
        parts=line.split('  ',1)
        if len(parts)!=2: errors.append(f'SHA256SUMS:{no}: invalid format'); continue
        expected,rel=parts; rel=rel.strip().replace('\\','/')
        p=(ROOT/rel).resolve()
        if ROOT not in p.parents or not p.is_file(): errors.append(f'SHA256SUMS:{no}: missing or escaping {rel}'); continue
        seen.add(rel)
        if digest(p)!=expected: errors.append(f'SHA256SUMS:{no}: hash mismatch {rel}')
    if 'bundle/packages/qwen38-flash-spark-3080-v2-config.tgz' not in seen: errors.append('package not covered by SHA256SUMS')
def verify_archives(errors):
    archives=sorted((ROOT/'bundle'/'packages').glob('*.tgz'))
    if not archives: errors.append('no .tgz package')
    for p in archives:
        try:
            with tarfile.open(p,'r:gz') as tar:
                members=tar.getmembers(); files=[m for m in members if m.isfile()]
                if not files: errors.append(f'empty archive {p.name}')
                for m in members:
                    if m.name.startswith('/') or '..' in Path(m.name).parts: errors.append(f'unsafe archive member {p.name}:{m.name}')
        except (OSError,tarfile.TarError) as exc: errors.append(f'invalid archive {p.name}: {exc}')
def verify_metrics(errors):
    try:
        m=json.loads((ROOT/'evidence/v2-metrics.json').read_text(encoding='utf-8'))
        if m['prefill_samples_tok_s'] != [1030.8,1163.4,1161.1]: errors.append('prefill samples changed')
        if m.get('prefill_peak_tok_s', max(m['prefill_samples_tok_s'])) != 1163.4: errors.append('prefill peak changed')
        if m['aggregate_decode_measured_tok_s'] != 242.37: errors.append('V2 aggregate decode changed')
        if m['single_stream_decode_tok_s'] != 62.0: errors.append('V2 single-stream decode changed')
        if m['solo_aggregate_decode_tok_s'] != 107.6: errors.append('solo aggregate baseline changed')
        if m['solo_single_stream_decode_tok_s'] != 22.3: errors.append('solo single-stream baseline changed')
        if m['decode_status'] != 'experimenter_reported_measurement': errors.append('decode source status missing')
        if m['decode_source']['raw_logs_attached'] is not False: errors.append('raw log disclosure missing')
        if 'raw logs' not in m['measurement_scope'].lower(): errors.append('measurement scope does not disclose raw logs')
    except Exception as exc: errors.append(f'metrics invalid: {exc}')
def scan_public_text(errors):
    roots=[ROOT/'README.md',ROOT/'README_ZH.md',ROOT/'NOTICE',ROOT/'Dockerfile',ROOT/'entrypoint.sh',ROOT/'apply-bundle.sh',ROOT/'run-dual-host.example.sh',ROOT/'bundle',ROOT/'evidence']
    for base in roots:
        paths=[base] if base.is_file() else list(base.rglob('*'))
        for p in paths:
            if not p.is_file() or p.name=='SHA256SUMS' or p.suffix.lower() not in TEXT_SUFFIXES: continue
            text=p.read_text(encoding='utf-8',errors='ignore')
            for marker in FORBIDDEN:
                if marker in text: errors.append(f'private marker {marker!r} in {p.relative_to(ROOT)}')
def main():
    errors=[]
    for rel in REQUIRED:
        if not (ROOT/rel).is_file(): errors.append(f'missing {rel}')
    verify_sums(errors); verify_archives(errors); verify_metrics(errors); scan_public_text(errors)
    if errors: print('FAIL'); print('\n'.join('- '+e for e in errors)); return 1
    print('PASS'); return 0
if __name__=='__main__': raise SystemExit(main())
