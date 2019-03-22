# -*- coding: utf-8 -*-

from setuptools import setup

with open("README.md", "r", encoding='utf8') as fh:
    long_description = fh.read()

setup(name='fcd_torch',
      version='1.0.7',
      author='',
      author_email='',
      description='Fréchet ChemNet Distance on PyTorch',
      url='https://github.com/insilicomedicine/FCD_torch',
      packages=['fcd_torch'],
      license='MIT',
      long_description=long_description,
      long_description_content_type="text/markdown",
      install_requires=[
          'torch',
          'numpy',
          'scipy',
      ],
      extras_require={
          'rdkit': ['rdkit'],
      },
      package_data={
        '': ['*.pt'],
      },
      include_package_data=True)
