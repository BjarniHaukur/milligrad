from setuptools import setup, find_packages

setup(
    name='milligrad',
    version='0.5.0',
    packages=find_packages(),
    install_requires=[
        'numpy>1.10.0',
    ],
)