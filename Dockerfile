FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

ARG NUM_BUILD_CORES
ENV VIRTUALGL_VERSION=3.1.2
# Enable all NVIDIA GPU capabilities (includes both CUDA and OpenGL)
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install all apt dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y tzdata
RUN apt-get update && apt-get install -y software-properties-common
# Add ppa for python install
RUN apt-add-repository -y ppa:deadsnakes/ppa
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    curl ca-certificates cmake git python3.10 python3.10-dev python3-pip \
    xvfb libxv1 libxrender1 libxrender-dev libgeos-dev \
    libboost-all-dev libcgal-dev ffmpeg python3-tk \
    libxtst6 libglu1-mesa libegl1 \
    texlive texlive-latex-extra dvipng cm-super \
    libeigen3-dev ninja-build wget

# Install VirtualGL
RUN curl -sSL https://github.com/VirtualGL/virtualgl/releases/download/"${VIRTUALGL_VERSION}"/virtualgl_"${VIRTUALGL_VERSION}"_amd64.deb -o virtualgl_"${VIRTUALGL_VERSION}"_amd64.deb && \
	dpkg -i virtualgl_*_amd64.deb && \
	/opt/VirtualGL/bin/vglserver_config -config +s +f -t && \
	rm virtualgl_*_amd64.deb

FROM base AS base-python
RUN pip3 install uv
COPY modules/requirements.txt requirements.txt
RUN uv pip install -r requirements.txt --system
RUN uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124 --system
RUN uv pip install torch_geometric -f https://data.pyg.org/whl/torch-2.5.0+cu124.html --system
RUN uv pip install sknw --system

# Build Spot
FROM base AS spot
# Install spot (for LTL specifications and PO-TLP)
RUN wget http://www.lrde.epita.fr/dload/spot/spot-2.12.tar.gz && \
    tar xvzf spot-2.12.tar.gz && rm spot-2.12.tar.gz && \
    cd spot-2.12 && \
    ./configure && \
    make -j8 2>&1 | tee make.log && make install


FROM base-python AS pkg-lsp
RUN mkdir /temp
COPY modules/lsp_accel modules/lsp_accel
RUN pip3 install modules/lsp_accel
COPY modules/lsp modules/lsp
RUN pip3 install modules/lsp
RUN cp -r /usr/local/lib/python3.10/dist-packages/lsp* /temp


FROM base-python AS pkg-core
RUN mkdir /temp
COPY modules/common modules/common
RUN pip3 install modules/common
RUN cp -r /usr/local/lib/python3.10/dist-packages/common* /temp
COPY modules/example modules/example
RUN pip3 install modules/example
RUN cp -r /usr/local/lib/python3.10/dist-packages/example* /temp
COPY modules/learning modules/learning
RUN pip3 install modules/learning
RUN cp -r /usr/local/lib/python3.10/dist-packages/learning* /temp
COPY modules/gridmap modules/gridmap
RUN pip3 install modules/gridmap
RUN cp -r /usr/local/lib/python3.10/dist-packages/gridmap* /temp


FROM base-python AS pkg-environments
RUN mkdir /temp
COPY modules/unitybridge modules/unitybridge
RUN pip3 install modules/unitybridge
RUN cp -r /usr/local/lib/python3.10/dist-packages/unitybridge* /temp
COPY modules/environments modules/environments
RUN pip3 install modules/environments
RUN cp -r /usr/local/lib/python3.10/dist-packages/environments* /temp
COPY modules/procthor modules/procthor
RUN pip3 install modules/procthor
RUN cp -r /usr/local/lib/python3.10/dist-packages/procthor* /temp


# Build the final image
FROM base-python AS target

# Needed for using matplotlib without a screen
RUN echo "backend: TkAgg" > matplotlibrc

# Copy and install the remaining code
COPY modules/conftest.py modules/conftest.py
COPY modules/setup.cfg modules/setup.cfg

# Migrate files from spot
COPY --from=spot /usr/local/lib/*spot* /usr/local/lib
COPY --from=spot /usr/local/lib/*bdd* /usr/local/lib
COPY --from=spot /usr/local/lib/python3.10/site-packages/spot /usr/local/lib/python3.10/site-packages/spot
COPY --from=spot /usr/local/lib/python3.10/site-packages/*buddy* /usr/local/lib/python3.10/site-packages/

# Migrate files from our package installs
COPY --from=pkg-core /temp/ /usr/local/lib/python3.10/dist-packages
COPY --from=pkg-core /modules/* /modules
COPY --from=pkg-lsp /temp/ /usr/local/lib/python3.10/dist-packages
COPY --from=pkg-lsp /modules/* /modules
COPY --from=pkg-environments /temp/ /usr/local/lib/python3.10/dist-packages
COPY --from=pkg-environments /modules/* /modules

# Set up the starting point for running the code
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
