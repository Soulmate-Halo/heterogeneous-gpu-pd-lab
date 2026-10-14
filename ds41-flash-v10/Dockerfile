FROM ubuntu:24.04

LABEL org.opencontainers.image.title="DeepSeek-V4.1-Flash V10 deployment bundle" \
      org.opencontainers.image.description="Deployment configs, launch templates and verification scripts for the V10 single-6000D + 4x Spark PD layout" \
      org.opencontainers.image.version="V10-20261014" \
      org.opencontainers.image.source="https://github.com/Soulmate-Halo/heterogeneous-gpu-pd-lab"

ENV DS41_V10_BUNDLE=/opt/ds41-flash-v10

COPY deploy/ ${DS41_V10_BUNDLE}/deploy/
COPY entrypoint.sh ${DS41_V10_BUNDLE}/entrypoint.sh
COPY README.md ${DS41_V10_BUNDLE}/README.md
COPY DEPLOYMENT.zh-CN.md ${DS41_V10_BUNDLE}/DEPLOYMENT.zh-CN.md
COPY verify_bundle.py ${DS41_V10_BUNDLE}/verify_bundle.py

RUN chmod 0755 ${DS41_V10_BUNDLE}/entrypoint.sh ${DS41_V10_BUNDLE}/deploy/*.sh \
    && ln -s ${DS41_V10_BUNDLE}/entrypoint.sh /usr/local/bin/ds41-v10

WORKDIR ${DS41_V10_BUNDLE}

ENTRYPOINT ["/opt/ds41-flash-v10/entrypoint.sh"]
CMD ["show-config"]
