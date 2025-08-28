from setuptools import setup, find_packages


setup(name='object_search_select',
      version='1.0.0',
      description='Model selection for object search in partially known environments',
      license="MIT",
      author='Abhishek Paudel',
      author_email='apaudel4@gmu.edu',
      packages=find_packages(),
      install_requires=['numpy', 'matplotlib'])
