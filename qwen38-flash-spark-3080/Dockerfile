FROM nvidia/cuda:13.0.2-devel-ubuntu24.04
ARG VCS_REF=local
LABEL org.opencontainers.image.title="Qwen3.8 Flash NVFP4 3080 + DGX Spark deployment kit" \
      org.opencontainers.image.description="Reproducible two-host Strata V2 runtime bundle; weights are mounted at runtime" \
      org.opencontainers.image.source="https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab" \
      org.opencontainers.image.revision="$VCS_REF" \
      org.opencontainers.image.version="V2" \
      org.opencontainers.image.metrics="decode_aggregate=242.37;solo_aggregate=107.60;decode_single=62.00;solo_single=22.30"
ENV BUNDLE_ROOT=/opt/qwen38-flash-spark-3080 \
    PATH=/opt/venv/bin:${PATH} \
    LD_LIBRARY_PATH=/opt/venv/lib/python3.12/site-packages/tvm_ffi/lib:${LD_LIBRARY_PATH} \
    STRATA_SGL_KERNEL_INCLUDE=/opt/deps/include \
    STRATA_TVM_FFI_INCLUDE=/opt/venv/lib/python3.12/site-packages/tvm_ffi/include \
    STRATA_TVM_FFI_LIB=/opt/venv/lib/python3.12/site-packages/tvm_ffi/lib/libtvm_ffi.so \
    STRATA_CACHE_DIR=/cache STRATA_STATE_DIR=/state STRATA_FOREGROUND=1 \
    BUILD_JOBS=4 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash ca-certificates coreutils python3 python3-venv python3-dev tar \
    build-essential cmake ninja-build git pkg-config libibverbs-dev libibverbs1 \
    librdmacm-dev librdmacm1 ibverbs-providers rdma-core libnuma-dev curl \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir apache-tvm-ffi==0.1.14.post1
COPY . ${BUNDLE_ROOT}/
RUN mkdir -p /opt/deps/include/sgl_kernel /cache /state \
    && cp ${BUNDLE_ROOT}/third_party/marlin/_sglkernel/*.h* /opt/deps/include/sgl_kernel/ \
    && test -f "${STRATA_TVM_FFI_INCLUDE}/tvm/ffi/extra/c_env_api.h" \
    && test -f "${STRATA_TVM_FFI_LIB}" \
    && chmod 0755 ${BUNDLE_ROOT}/entrypoint.sh ${BUNDLE_ROOT}/apply-bundle.sh ${BUNDLE_ROOT}/run-dual-host.example.sh ${BUNDLE_ROOT}/scripts/*.sh \
    && python3 ${BUNDLE_ROOT}/verify_bundle.py
WORKDIR ${BUNDLE_ROOT}
HEALTHCHECK --interval=30s --timeout=5s --start-period=30m --retries=3 CMD bash scripts/container-health.sh
ENTRYPOINT ["/opt/qwen38-flash-spark-3080/entrypoint.sh"]
CMD ["help"]
