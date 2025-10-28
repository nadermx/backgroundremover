FROM continuumio/miniconda3:23.5.2-0 AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      apt-transport-https \
      bash \
      build-essential \
      git

RUN conda install 'ffmpeg>=4.4.0' -c conda-forge

WORKDIR /usr/local/src
COPY . .

# Install PyTorch and other dependencies via pip
# This ensures compatible versions are installed together as specified in requirements.txt
RUN pip --no-cache-dir -v install .

# optimize layers
FROM debian:bullseye-slim
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

# Create directory for model cache
RUN mkdir -p /root/.u2net

# Define volume for model persistence
# To use: docker run -v ~/.u2net:/root/.u2net ...
VOLUME ["/root/.u2net"]

WORKDIR /tmp
ENTRYPOINT ["python", "-m", "backgroundremover.cmd.cli"]