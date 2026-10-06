#!/usr/bin/env python3
"""Check deployment prerequisites without reading any model or credential data."""
import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

DEFAULTS = {
    'STRATA_PACK_HOT': '/models/strata-pack-hot',
    'STRATA_PACK_COLD': '/models/strata-cold-pack',
    'STRATA_DENSE_GGUF': '/models/dense.gguf',
    'STRATA_PLE_FP8': '/models/ple-fp8.bin',
    'STRATA_PROFILE': '/models/profile.bin',
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', nargs='?', choices=['spark', '3080'])
    role = parser.parse_args().role
    errors = []
    def check(condition, message):
        print(('PASS ' if condition else 'MISSING ') + message)
        if not condition:
            errors.append(message)

    print('Topology: Spark solo Prefill; RDMA hot/cold expert joint Decode.')
    print('LIMIT: frozen router has no cross-host KV export/import; no seamless handoff.')
    if not role:
        print('General diagnostics only; use doctor spark or doctor 3080 for strict readiness.')
    check(platform.system() == 'Linux', 'Linux host/container')
    for command in ['python3', 'cmake', 'nvcc']:
        check(shutil.which(command) is not None, command + ' available')
    if role:
        machine = platform.machine().lower()
        check(machine in (['aarch64', 'arm64'] if role == 'spark' else ['x86_64', 'amd64']), 'CPU architecture matches ' + role)
        names = ['STRATA_DENSE_GGUF', 'STRATA_PLE_FP8', 'STRATA_PACK_COLD' if role == 'spark' else 'STRATA_PACK_HOT']
        if role == '3080':
            names.append('STRATA_PROFILE')
        for name in names:
            path = Path(os.environ.get(name) or DEFAULTS[name])
            exists = path.is_dir() if name in ['STRATA_PACK_COLD', 'STRATA_PACK_HOT'] else path.is_file()
            check(exists, name + ' exists (mount the matching asset under /models)')
        if role == '3080':
            peer = os.environ.get('STRATA_REMOTE_HOST', '')
            check(bool(peer) and not peer.endswith('.example') and '<' not in peer, 'STRATA_REMOTE_HOST configured for Spark RDMA')
        check(Path('/dev/infiniband').is_dir(), 'RDMA /dev/infiniband mounted')
        check(shutil.which('nvidia-smi') is not None, 'GPU runtime mounted (NVIDIA Container Toolkit)')
        if shutil.which('nvidia-smi'):
            try:
                caps = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], text=True, timeout=15).splitlines()
                check(('12.1' if role == 'spark' else '8.6') in [x.strip() for x in caps], 'GPU compute capability matches ' + role)
            except (OSError, subprocess.SubprocessError):
                check(False, 'GPU compute capability could not be queried')
    if errors:
        print('Read README_ZH.md / ASSETS.md for preparation. Model assets are not included in this image.')
    if role and errors:
        return 2
    print('DOCTOR_GENERAL_OK' if not role else 'DOCTOR_ROLE_OK')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
