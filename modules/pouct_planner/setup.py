from setuptools import setup, find_packages


setup(name='pouct_planner',
      version='1.0.0',
      description='Po-UCT Planner using MCTS',
      license="MIT",
      author='Gregory J. Stein',
      author_email='gjstein@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])
