FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04

ENV VIRTUALGL_VERSION 2.5.2
ARG NUM_BUILD_CORES

# Install all apt dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get update && apt-get install -y software-properties-common
# Add ppa for python3.8 install
RUN apt-add-repository -y ppa:deadsnakes/ppa
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl ca-certificates cmake git python3.8 python3.8-dev \
    xvfb libxv1 libxrender1 libxrender-dev gcc-10 g++-10 libgeos-dev \
    libboost-all-dev libcgal-dev ffmpeg python3.8-tk \
    texlive texlive-latex-extra dvipng cm-super \
    libeigen3-dev ninja-build wget

# Install VirtualGL
RUN curl -sSL https://downloads.sourceforge.net/project/virtualgl/"${VIRTUALGL_VERSION}"/virtualgl_"${VIRTUALGL_VERSION}"_amd64.deb -o virtualgl_"${VIRTUALGL_VERSION}"_amd64.deb && \
	dpkg -i virtualgl_*_amd64.deb && \
	/opt/VirtualGL/bin/vglserver_config -config +s +f -t && \
	rm virtualgl_*_amd64.deb

# Install python dependencies
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm get-pip.py
COPY modules/requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
# RUN pip3 install torch==2.0.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip3 install torch
# RUN pip install torch_geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
# RUN pip3 install captum

# Needed for using matplotlib without a screen
RUN echo "backend: TkAgg" > matplotlibrc

# Copy and install the remaining code
COPY modules/conftest.py modules/conftest.py
COPY modules/setup.cfg modules/setup.cfg

COPY modules/lsp_accel modules/lsp_accel
RUN pip3 install modules/lsp_accel

COPY modules/common modules/common
RUN pip3 install modules/common
COPY modules/example modules/example
RUN pip3 install modules/example
COPY modules/learning modules/learning
RUN pip3 install modules/learning
COPY modules/gridmap modules/gridmap
RUN pip3 install modules/gridmap
COPY modules/unitybridge modules/unitybridge
RUN pip3 install modules/unitybridge
COPY modules/environments modules/environments
RUN pip3 install modules/environments
COPY modules/lsp modules/lsp
RUN pip3 install modules/lsp

# Set up the starting point for running the code
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
