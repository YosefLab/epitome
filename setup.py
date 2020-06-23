from setuptools import find_packages, setup
from version import version as this_version

try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

import os
import sys
import subprocess

# Utility function to read the README file.
# Used for the long_description.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements.txt', session='hack')

try:
    reqs = [str(ir.req) for ir in install_reqs]
except: # pip > 20
    reqs = [str(ir.requirement) for ir in install_reqs]

# append tensorflow or tensorflow-gpu to reqs
# need nightly build to work with tensorflow probability
TENSORFLOW_VERSION="2.1.0"

try:
    subprocess.check_output(["nvidia-smi", "-L"])
    tf_req = "tensorflow-gpu==%s" % TENSORFLOW_VERSION
except:
    tf_req = "tensorflow==%s" % TENSORFLOW_VERSION

reqs.append(tf_req)

# Cython must be installed before pyranges
err = subprocess.call(["pip","install","cython"])
if err != 0:
    raise Exception("Installing cython failed with error %i" % err)

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='epitome',
    version=this_version,
    description='ML model for predicting ChIP-seq peaks in new cell types from ENCODE cell lines',
    author='Alyssa Kramer Morrow',
    author_email='akmorrow@berkeley.edu',
    project_urls={
        'Documentation': 'https://readthedocs.org', # TODO
        'Source': 'https://github.com/akmorrow13/epitome'
    },
    classifiers=[
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        # Pick your license as you wish (should match "license" above)
         'License :: OSI Approved :: Apache Software License',
        # Python versions supported
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    license="Apache License, Version 2.0",
    keywords='ENCODE ChIP-seq_peaks prediction histone transcription_factor',
    install_requires=reqs,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['*.test.*']),
    python_requires='>=3'
)
 
