FROM debian:bookworm-slim
ARG VCS_REF=local
LABEL org.opencontainers.image.title="Qwen3.8 Flash NVFP4 3080 + DGX Spark deployment kit" \
      org.opencontainers.image.description="Reproducible two-host Strata V2 runtime bundle; weights are mounted at runtime" \
      org.opencontainers.image.source="https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab" \
      org.opencontainers.image.revision="$VCS_REF" \
      org.opencontainers.image.version="V2" \
      org.opencontainers.image.metrics="prefill_peak=1163.40;decode_measured=60.25;decode_v2_projected=240.00"
ENV BUNDLE_ROOT=/opt/qwen38-flash-spark-3080
RUN apt-get update && apt-get install -y --no-install-recommends bash ca-certificates coreutils python3 python3-venv tar && rm -rf /var/lib/apt/lists/*
COPY . ${BUNDLE_ROOT}/
RUN chmod 0755 ${BUNDLE_ROOT}/entrypoint.sh ${BUNDLE_ROOT}/apply-bundle.sh ${BUNDLE_ROOT}/run-dual-host.example.sh ${BUNDLE_ROOT}/scripts/*.sh
WORKDIR ${BUNDLE_ROOT}
ENTRYPOINT ["/opt/qwen38-flash-spark-3080/entrypoint.sh"]
CMD ["verify"]
