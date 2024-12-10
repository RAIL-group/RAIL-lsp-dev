from subprocess import sys, call

packages = ["numpy==2.0.2", "scipy", "matplotlib", "bpycv", "shapely"]
call([sys.executable, "-m", "ensurepip"])
call([sys.executable, "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])
call([sys.executable, "-m", "pip", "install"] + packages)
