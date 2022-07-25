from setuptools import find_packages, setup
setup(
    name='dnn2bnn',
    packages=find_packages(include=['dnn2bnn', 'dnn2bnn.mappers', 'dnn2bnn.models', 'dnn2bnn.metrics']),
    version='0.1.0',
    description='Library to translate DNN into BNN',
    author='Victoria Cal',
    license='MIT'
)