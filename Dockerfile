FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 AS base

ENV VIRTUALGL_VERSION=3.1.2
# Enable all NVIDIA GPU capabilities (includes both CUDA and OpenGL)
ENV NVIDIA_DRIVER_CAPABILITIES=all
ENV VIRTUAL_ENV=/opt/.venv
ENV PYTHON_VERSION=3.10

# Install all apt dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    software-properties-common \
    curl ca-certificates cmake git \
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

# Install uv & Initialize python setup
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv venv /opt/.venv --python ${PYTHON_VERSION}
RUN uv pip install pybind11 wheel setuptools


FROM base AS base-python
RUN uv pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 --index-url https://download.pytorch.org/whl/cu124
RUN uv pip install torch_geometric -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
COPY modules/requirements.txt requirements.txt
RUN uv pip install -r requirements.txt
RUN uv pip install sknw

# Install PDDLStream
RUN git clone https://github.com/caelan/pddlstream.git \
	&& cd pddlstream && ls -a && cat .gitmodules\
	&& sed -i 's/ss-pybullet/pybullet-planning/' .gitmodules \
	&& sed -i 's/git@github.com:caelan\/downward.git/https:\/\/github.com\/caelan\/downward/' .gitmodules \
	&& git submodule update --init --recursive
RUN cd pddlstream\
	&& ./downward/build.py
ENV PYTHONPATH="/pddlstream:${PYTHONPATH}"


FROM base AS spot
# Install spot (for LTL specifications and PO-TLP)
RUN wget http://www.lrde.epita.fr/dload/spot/spot-2.12.tar.gz && \
    tar xvzf spot-2.12.tar.gz && rm spot-2.12.tar.gz && \
    cd spot-2.12 && \
    ./configure && \
    make -j8 2>&1 | tee make.log && make install


FROM base AS pkg-lsp
COPY modules/lsp_accel modules/lsp_accel
COPY modules/lsp modules/lsp
COPY modules/lsp_xai modules/lsp_xai
RUN uv pip install modules/* --no-build-isolation


FROM base AS pkg-core
COPY modules/common modules/common
COPY modules/example modules/example
COPY modules/learning modules/learning
COPY modules/gridmap modules/gridmap
RUN uv pip install modules/*


FROM base AS pkg-environments
COPY modules/unitybridge modules/unitybridge
COPY modules/environments modules/environments
COPY modules/procthor modules/procthor
COPY modules/taskplan modules/taskplan
COPY modules/object_search modules/object_search
COPY modules/procint modules/procint
RUN uv pip install modules/*


FROM base-python AS target

# Needed for using matplotlib without a screen
RUN echo "backend: TkAgg" > matplotlibrc

# Copy and install the remaining code
COPY modules/conftest.py modules/conftest.py
COPY modules/setup.cfg modules/setup.cfg

# # Migrate files from our package installs
COPY --from=pkg-environments /opt/.venv/lib/ /opt/.venv/lib/
COPY --from=pkg-environments /modules /modules
COPY --from=pkg-core /opt/.venv/lib/ /opt/.venv/lib/
COPY --from=pkg-core /modules /modules
COPY --from=pkg-lsp /opt/.venv/lib/ /opt/.venv/lib/
COPY --from=pkg-lsp /modules /modules

# Migrate files from spot
COPY --from=spot /usr/local/lib/*spot* /opt/.venv/lib
COPY --from=spot /usr/local/lib/*bdd* /opt/.venv/lib
COPY --from=spot /usr/local/lib/python3.10/site-packages/spot /opt/.venv/lib/python${PYTHON_VERSION}/site-packages/spot
COPY --from=spot /usr/local/lib/python3.10/site-packages/*buddy* /opt/.venv/lib/python${PYTHON_VERSION}/site-packages/

# Set up the starting point for running the code
COPY entrypoint.sh /entrypoint.sh
RUN chmod 755 /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
