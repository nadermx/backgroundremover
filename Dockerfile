FROM continuumio/miniconda3:23.5.2-0 as builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      apt-transport-https \
      bash \
      build-essential \
      git

RUN conda install 'ffmpeg>=4.4.0' -c conda-forge
RUN conda install pytorch torchvision torchaudio cpuonly -c pytorch

WORKDIR /usr/local/src
COPY . .

RUN pip --no-cache-dir -v install .

# optimize layers
FROM debian:bullseye-slim
COPY --from=builder /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:$PATH

WORKDIR /tmp
ENTRYPOINT ["python", "-m", "backgroundremover.cmd.cli"]