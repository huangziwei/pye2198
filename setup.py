from setuptools import setup, find_packages

setup(name='pye2198',
      version='0.0.1',
      description='A Python toolkit for processing e2198 dataset',
      author='Ziwei Huang',
      author_email='huang-ziwei@outlook.com',
      url='https://github.com/huangziwei/pye2198',
      packages=find_packages(),
      install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "matplotlib-scalebar",
        "networkx",
      ],
     )
