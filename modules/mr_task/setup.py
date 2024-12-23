from setuptools import setup, find_packages


setup(name='mr_task',
      version='1.0.0',
      description='Multi-robot Task Planning module',
      license="MIT",
      author='Gregory J. Stein',
      author_email='gjstein@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])
