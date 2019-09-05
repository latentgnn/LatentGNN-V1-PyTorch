#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   setup.py
@Time    :   2019/05/27 10:05:01
@Author  :   Songyang Zhang 
@Version :   1.0
@Contact :   sy.zhangbuaa@hotmail.com
@License :   (C)Copyright 2019-2020, PLUS Group@ShanhaiTech University
@Desc    :   None
'''


import glob
import os
import subprocess
import io

from setuptools import find_packages, setup
import setuptools.command.develop
import setuptools.command.install

cwd = os.path.dirname(os.path.abspath(__file__))

version = '1.0.0'

try:
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    version += '+' + sha[:7]
except Exception:
    pass

def create_version_file():
    global version, cwd 
    print('---- Building version ' + version)
    version_path = os.path.join(cwd, 'lib', 'version.py')
    with open(version_path, 'w') as f:
        f.write('"""This is latentgnn version file"""\n')
        f.write("__version__ = '{}'\n".format(version))

class install(setuptools.command.install.install):
    def run(self):
        create_version_file()
        setuptools.command.install.install.run(self)

class develop(setuptools.command.develop.develop):
    def run(self):
        create_version_file()
        setuptools.command.develop.develop.run(self)


requirments = [
    'torch>=1.0.0', 
    'numpy',
    'torchvision']

setup(
    name = 'latentgnn',
    version = version,
    author = 'Songyang Zhang',
    author_email = 'sy.zhangbuaa@gmail.com',
    url = 'https://github.com/latentgnn/LatentGNN-V1-PyTorch',
    description = 'Latent Graph Neural Network -- PyTorch',
    license = 'MIT',
    install_requires = requirments,
    cmdclass = {
        'install': install,
        'develop': develop,
    }
)
