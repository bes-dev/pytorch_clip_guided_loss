"""
Copyright 2021 by Sergei Belousov

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from setuptools import setup, find_packages
try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

readme = open('README.md').read()

VERSION = '2021.12.02.0'

install_reqs = parse_requirements("requirements.txt")
requirements = [str(ir.req) for ir in install_reqs]

setup(
    # Metadata
    name='pytorch_clip_guided_loss',
    version=VERSION,
    author='Sergei Belousov aka BeS',
    author_email='sergei.o.belousov@gmail.com',
    description='Pytorch implementation of the CLIP guided loss.',
    long_description=readme,
    long_description_content_type='text/markdown',

    # Package info
    packages=find_packages(exclude=('*test*',)),

    #
    zip_safe=True,
    install_requires=requirements,

    # Classifiers
    classifiers=[
        'Programming Language :: Python :: 3',
    ],


    # install .json configs
    package_data={
        "pytorch_clip_guided_loss": ["configs/*.yml"]
    },
    include_package_data=True
)
