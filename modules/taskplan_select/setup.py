from setuptools import setup, find_packages


setup(name='taskplan_select',
      version='1.0.0',
      description='Model selection for task planning under uncertainty',
      license="MIT",
      author='Abhishek Paudel',
      author_email='apaudel4@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])
