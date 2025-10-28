FROM continuumio/miniconda3:23.5.2-0 as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      apt-transport-https \
      bash \
      build-essential \
      git

RUN conda install 'ffmpeg>=4.4.0' -c conda-forge
# Install PyTorch - torchaudio is not required for backgroundremover
# On ARM64 (Apple Silicon), torchaudio may not be available, so we skip it
RUN conda install pytorch torchvision cpuonly -c pytorch || \
    conda install pytorch torchvision -c pytorch

WORKDIR /usr/local/src
COPY . .

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