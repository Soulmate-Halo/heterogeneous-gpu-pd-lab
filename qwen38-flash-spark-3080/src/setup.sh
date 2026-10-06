#!/bin/sh
# Strata for Linux: the first run installs everything and starts the model; later runs just start it.
# Needs only an NVIDIA driver (or, experimental: an AMD RX 7900 XT/XTX with the amdgpu driver, see docs/AMD_HIP.md).
# Python (with venv) is installed through apt/dnf if it is missing (asks for sudo).
cd "$(dirname "$0")" || exit 1
# Python 3.10+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (that is the separate
# python3-venv package), and a venv made without it has no pip
ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
# a .venv from an earlier run that failed half-way has a python but no pip: start it again
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3 python; do
    if command -v $c >/dev/null 2>&1 && ok_py $c; then
      PY=$c; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.10+ with venv is needed; installing it (sudo will ask for your password) ..."
    if command -v apt-get >/dev/null 2>&1; then
      sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip
    elif command -v dnf >/dev/null 2>&1; then
      sudo dnf install -y python3 python3-pip
    elif command -v pacman >/dev/null 2>&1; then
      sudo pacman -S --noconfirm python python-pip
    fi
    PY=python3
    if ! ok_py $PY; then
      echo "Please install Python 3.10 or newer with venv (Ubuntu/Debian: sudo apt install python3-venv), then run"
      echo "./setup.sh again."
      exit 1
    fi
  fi
  # a private environment inside this folder (system Python stays untouched; newer distros refuse global pip)
  $PY -m venv .venv || { rm -rf .venv; echo "could not create .venv: sudo apt install python3-venv"; exit 1; }
fi
exec .venv/bin/python setup.py "$@"
