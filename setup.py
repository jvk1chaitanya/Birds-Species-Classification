from setuptools import setup, find_packages
from typing import List

def find_reqiurements(file_name: str) -> List[str]:
    req = []
    with open(file_name) as f:
        req = f.read().splitlines()
        req = [r.replace('\n','') for r in req]
    if '-e .' in req:
        req.remove('-e .')
    return req

setup(
    name='birds-classification-model',
    version='0.01',
    description='A package to classify birds images',
    author='Jvk Chaitanya',
    author_email='jvkchaitanya123@gmail.com',
    packages=find_packages(),
    install_requires=find_reqiurements('requirements.txt')   
)