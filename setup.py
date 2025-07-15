# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 07:49:16 2017

@author: Ali Aghaeifar
"""


from setuptools import setup


setup(name='recomrzero', # this will be name of package in packages list : pip list 
      version='0.2.0',
      description='Reconstruction utilities for MRzero data',
      keywords='reconstruction,mri,nifti',
      author='Cyprien Bouton',
      license='MIT License',
      packages=['reco_mrzero'],
      install_requires = [
            'MRzeroCore',
            'numpy',
            'torch',
            'nibabel',
            'ggrappa @ git+https://github.com/CyprienBouton/ggrappa.git@fix_grid_size',
      ]
     )
