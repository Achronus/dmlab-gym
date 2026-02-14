"""Setup for the deepmind_lab module."""

import setuptools

setuptools.setup(
    name='deepmind-lab',
    version='1.0',
    description='DeepMind Lab: A 3D learning environment',
    long_description='',
    url='https://github.com/deepmind/lab',
    author='DeepMind',
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy >= 1.26',
    ],
    extras_require={
        'dmenv_module': ['dm-env'],
    },
    include_package_data=True)
