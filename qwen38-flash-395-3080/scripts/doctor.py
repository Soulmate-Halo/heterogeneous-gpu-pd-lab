#!/usr/bin/env python3
"""Check single-host 395 + RTX 3080 deployment prerequisites without reading weights."""
import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path

DEFAULTS = {
    'STRATA_ENGINE_BIN': '/models/strata/strata',
    'STRATA_EXPERTS_PACK': '/models/experts.bin',
    'STRATA_DENSE_GGUF': '/models/dense.gguf',
    'STRATA_PLE': '/models/ple.bin',
    'STRATA_EMBED': '/models/embedding.bin',
    'STRATA_MTP': '/models/mtp.bin',
}
MIN_MEMORY_GB = 96


def memory_gb():
    try:
        for line in Path('/proc/meminfo').read_text(encoding='utf-8').splitlines():
            if line.startswith('MemTotal:'):
                return int(line.split()[1]) / (1024 * 1024)
    except OSError:
        return None
    return None


def cpu_has_avx512():
    try:
        for line in Path('/proc/cpuinfo').read_text(encoding='utf-8').splitlines():
            if line.startswith('flags') and 'avx512f' in line.split(':', 1)[1].split():
                return True
    except OSError:
        return False
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', nargs='?', choices=['engine', '395', '3080'])
    role = parser.parse_args().role
    errors = []

    def check(condition, message):
        print(('PASS ' if condition else 'MISSING ') + message)
        if not condition:
            errors.append(message)

    print('Topology: single host; RTX 3080 (hot experts + dense path) with AI Max 395 + 8060S cold experts.')
    print('LIMIT: cold experts stay on the 395/CPU pool; no cross-host KV handoff.')
    check(platform.system() == 'Linux', 'Linux host/container')
    check(platform.machine().lower() in ('x86_64', 'amd64'), 'CPU architecture is x86_64')
    check(cpu_has_avx512(), 'CPU supports AVX-512 (avx512f in /proc/cpuinfo)')
    total = memory_gb()
    check(total is not None and total >= MIN_MEMORY_GB,
          'system memory >= %d GB (found %s)' % (MIN_MEMORY_GB, 'unknown' if total is None else '%.1f GB' % total))
    check(shutil.which('nvidia-smi') is not None, 'nvidia-smi available (NVIDIA Container Toolkit)')
    if shutil.which('nvidia-smi'):
        try:
            caps = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader'], text=True, timeout=15)
            check('8.6' in [x.strip() for x in caps.splitlines()], 'GPU compute capability 8.6 (RTX 3080)')
        except (OSError, subprocess.SubprocessError):
            check(False, 'GPU compute capability could not be queried')
    for name in ['STRATA_EXPERTS_PACK', 'STRATA_DENSE_GGUF', 'STRATA_PLE', 'STRATA_EMBED', 'STRATA_MTP', 'STRATA_ENGINE_BIN']:
        path = Path(os.environ.get(name) or DEFAULTS[name])
        check(path.is_file(), name + ' exists (mount the matching asset under /models)')
    hip_available = shutil.which('rocminfo') is not None or Path('/opt/rocm').is_dir()
    if hip_available:
        print('PASS ROCm/HIP detected: set STRATA_HIPDECODE=1 to offload cold experts to the 8060S')
    else:
        print('INFO no ROCm/HIP found: cold experts run on the Zen5 AVX-512 CPU pool (STRATA_HIPDECODE=0)')
    if errors:
        print('Read README_ZH.md for preparation. Model weights and expert packs are not included in this image.')
        if role:
            return 2
    print('DOCTOR_ROLE_OK' if role else 'DOCTOR_GENERAL_OK')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
