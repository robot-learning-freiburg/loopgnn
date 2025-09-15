from setuptools import setup

setup(
   name='loopgnn',
   author='Martin Buechner',
   author_email='buechner@cs.uni-freiburg.de',
   packages=['loopgnn',
             'loopgnn.data',
             'loopgnn.models',
             'loopgnn.utils',
             'loopgnn.scripts',
             ],
   license='LICENSE.md',
   description='Robust visual loop closure detection with graph neural networks',
)
