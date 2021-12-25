# Dockerfile for building the container available also from dockerhub.



FROM python:3.7-slim
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes
    
RUN conda install numpy scikit-learn pandas pyyaml bzip2 pystan rasterio shapely fiona
RUN apt-get update && apt-get install -y git procps git-lfs
ENV REPO_NAME=vegcover-manuscript-12_2021
ENV HOME_DIR=/home/vegcover
ARG HOST_DATE
RUN echo Cloning repo on $HOST_DATE && mkdir $HOME_DIR && cd $HOME_DIR && git clone https://github.com/luotsi/$REPO_NAME.git
RUN cd $HOME_DIR/$REPO_NAME && git lfs pull
WORKDIR $HOME_DIR/$REPO_NAME
ENV PATH "$PATH:$HOME_DIR/$REPO_NAME"

