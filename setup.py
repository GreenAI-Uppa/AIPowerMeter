from distutils.core import setup
setup(name='foo',
      version='1.0',
      packages=['power_consumption_measure'],
      )
install_requires=[
       'psutil',
       'pandas',
       'numpy',
       'networkx',
   ],
