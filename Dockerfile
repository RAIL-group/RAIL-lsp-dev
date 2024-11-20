FROM nvidia/cudagl:11.3.1-devel-ubuntu20.04 AS base

ENV VIRTUALGL_VERSION=2.5.2
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

FROM base AS base-python
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py && rm get-pip.py
COPY modules/requirements.txt requirements.txt
RUN pip3 install uv
RUN pip3 install lit==15.0.7  # needed to help UV install torch
RUN uv pip install -r requirements.txt --system
RUN uv pip install torch==2.0.0+cu118 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --system
RUN pip install torch_geometric -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# Build Spot
FROM base AS spot
# Use gcc-10 and g++-10
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10 && \
	update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 10
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
RUN cp -r /usr/local/lib/python3.8/dist-packages/lsp* /temp


FROM base-python AS pkg-core
RUN mkdir /temp
COPY modules/common modules/common
RUN pip3 install modules/common
RUN cp -r /usr/local/lib/python3.8/dist-packages/common* /temp
COPY modules/example modules/example
RUN pip3 install modules/example
RUN cp -r /usr/local/lib/python3.8/dist-packages/example* /temp
COPY modules/learning modules/learning
RUN pip3 install modules/learning
RUN cp -r /usr/local/lib/python3.8/dist-packages/learning* /temp
COPY modules/gridmap modules/gridmap
RUN pip3 install modules/gridmap
RUN cp -r /usr/local/lib/python3.8/dist-packages/gridmap* /temp


FROM base-python AS pkg-environments
RUN mkdir /temp
COPY modules/unitybridge modules/unitybridge
RUN pip3 install modules/unitybridge
RUN cp -r /usr/local/lib/python3.8/dist-packages/unitybridge* /temp
COPY modules/environments modules/environments
RUN pip3 install modules/environments
RUN cp -r /usr/local/lib/python3.8/dist-packages/environments* /temp


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
COPY --from=spot /usr/local/lib/python3.8/site-packages/spot /usr/local/lib/python3.8/site-packages/spot
COPY --from=spot /usr/local/lib/python3.8/site-packages/*buddy* /usr/local/lib/python3.8/site-packages/

# Migrate files from our package installs
COPY --from=pkg-core /temp/ /usr/local/lib/python3.8/dist-packages
COPY --from=pkg-core /modules/* /modules
COPY --from=pkg-lsp /temp/ /usr/local/lib/python3.8/dist-packages
COPY --from=pkg-lsp /modules/* /modules
COPY --from=pkg-environments /temp/ /usr/local/lib/python3.8/dist-packages
COPY --from=pkg-environments /modules/* /modules

# Set up the starting point for running the code
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
