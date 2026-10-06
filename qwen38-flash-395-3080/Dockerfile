FROM nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG VCS_REF=local
LABEL org.opencontainers.image.title="Qwen3.8 Flash-Next NVFP4 395 + RTX 3080 single-host deployment kit" \
      org.opencontainers.image.description="Reproducible single-host hot/cold expert split; weights and packs are mounted at runtime" \
      org.opencontainers.image.source="https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab" \
      org.opencontainers.image.revision="$VCS_REF" \
      org.opencontainers.image.version="FLASH-395-01" \
      org.opencontainers.image.metrics="prefill=800+;decode=40+;baseline_prefill=273.04;baseline_decode=41.33"
ENV BUNDLE_ROOT=/opt/qwen38-flash-395-3080 \
    PATH=/opt/venv/bin:${PATH} \
    STRATA_CACHE_DIR=/cache STRATA_STATE_DIR=/state STRATA_FOREGROUND=1 \
    BUILD_JOBS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates coreutils python3 python3-venv python3-dev tar \
    build-essential cmake ninja-build git pkg-config libnuma-dev curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv
COPY . ${BUNDLE_ROOT}/
# sm_86 (RTX 3080) is the only GPU architecture baked into this image. ROCm/HIP is
# deliberately NOT installed here: the AI Max 395 (8060S, gfx1151) hip library is
# compiled at container start with hipcc --offload-arch=gfx1151 and cached under
# /cache (see scripts/container-start.sh), so the image stays linux/amd64 + CUDA only.
RUN mkdir -p /cache /state \
    && chmod 0755 ${BUNDLE_ROOT}/entrypoint.sh ${BUNDLE_ROOT}/apply-bundle.sh ${BUNDLE_ROOT}/scripts/*.sh \
    && python3 ${BUNDLE_ROOT}/verify_bundle.py
WORKDIR ${BUNDLE_ROOT}
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 CMD bash scripts/health.sh
ENTRYPOINT ["/opt/qwen38-flash-395-3080/entrypoint.sh"]
CMD ["help"]
