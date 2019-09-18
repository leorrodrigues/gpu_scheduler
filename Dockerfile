FROM scheduler
LABEL maintainer "Leonardo Rosa Rodrigues"

RUN cd /gpu_scheduler/ && rm -rf build bin && git pull
