from subprocess import sys, call

packages = ['numpy==1.24.4', 'scipy', 'matplotlib']
call([sys.executable,'-m','ensurepip'])
call([sys.executable, '-m', 'pip', 'install', '-U', 'pip', 'setuptools', 'wheel'])
call([sys.executable, '-m', 'pip', 'install'] + packages)
