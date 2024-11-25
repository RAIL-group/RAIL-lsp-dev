from subprocess import sys, call

packages = ['numpy', 'scipy', 'matplotlib']
call([sys.executable,'-m','ensurepip'])
call([sys.executable, '-m', 'pip', 'install', '-U', 'pip', 'setuptools', 'wheel'])
call([sys.executable, '-m', 'pip', 'install'] + packages)
