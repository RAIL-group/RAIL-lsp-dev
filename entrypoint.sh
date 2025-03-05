#!/bin/bash
set -e

# Needed to point the system towards pytorch CUDA
export LD_LIBRARY_PATH=/opt/venv/lib/python3.10/site-packages/torch/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/venv/lib:$LD_LIBRARY_PATH

# Ensures 'spot' is on the path
export PYTHONPATH=/opt/venv/lib/python3.10/site-packages/:$PYTHONPATH


# Main command
if [ "$XPASSTHROUGH" = true ]
then
    echo "Passing through local X server."
    $@
elif nvidia-smi > /dev/null 2>&1 ; then
    echo "Using Docker virtual X server (with GPU)."
    export VGL_DISPLAY=$DISPLAY
    xvfb-run -a --server-num=$((99 + $RANDOM % 10000)) \
	     --server-args='-screen 0 640x480x24 +extension GLX +render -noreset' vglrun $@
else
    echo "Using Docker virtual X server (no GPU)."
    xvfb-run -a --server-num=$((99 + $RANDOM % 10000)) \
	     --server-args='-screen 0 640x480x24' $@
fi
