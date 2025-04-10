from setuptools import setup, find_packages

setup(
    name="procgraph_intervention",
    version="1.0.0",
    description="Intervention learning for task planning in ProcTHOR",
    license="MIT",
    author="Raihan Islam Arnob",
    author_email="rarnob@gmu.edu",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib"],
)
