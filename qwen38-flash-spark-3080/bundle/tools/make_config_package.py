#!/usr/bin/env python3
"""Rebuild the deterministic public configuration overlay."""
from pathlib import Path
import gzip, hashlib, io, tarfile
ROOT=Path(__file__).resolve().parents[2]; BUNDLE=ROOT/'bundle'; OUT=BUNDLE/'packages'/'qwen38-flash-spark-3080-v2-config.tgz'
NAMES={'config/runtime.env':BUNDLE/'runtime.env','config/engine-3080.serve.json':BUNDLE/'config/engine-3080.serve.json','config/worker-spark.json':BUNDLE/'config/worker-spark.json','config/router.json':BUNDLE/'config/router.json','scripts/run-3080.sh':ROOT/'scripts'/'run-3080.sh','scripts/run-spark.sh':ROOT/'scripts'/'run-spark.sh','scripts/health.sh':ROOT/'scripts'/'health.sh','scripts/stop.sh':ROOT/'scripts'/'stop.sh','scripts/container-help.sh':ROOT/'scripts'/'container-help.sh','scripts/container-health.sh':ROOT/'scripts'/'container-health.sh','scripts/container-smoke.sh':ROOT/'scripts'/'container-smoke.sh','scripts/doctor.py':ROOT/'scripts'/'doctor.py'}
data={n:p.read_bytes() for n,p in NAMES.items()}; data['README.txt']=b'Qwen3.8 Flash-Next V2 runtime overlay. Mount model assets separately.\n'; data['manifest.txt']=('# deterministic V2 overlay\n'+'\n'.join(f'{n}  {hashlib.sha256(v).hexdigest()}  {len(v)}' for n,v in sorted(data.items()))+'\n').encode()
buf=io.BytesIO()
with gzip.GzipFile(fileobj=buf,mode='wb',mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode='w',format=tarfile.PAX_FORMAT) as tar:
    for n,v in sorted(data.items()):
      i=tarfile.TarInfo(n); i.size=len(v); i.mtime=0; i.uid=i.gid=0; i.mode=0o755 if n.endswith('.sh') else 0o644; tar.addfile(i,io.BytesIO(v))
OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_bytes(buf.getvalue()); print(OUT)
