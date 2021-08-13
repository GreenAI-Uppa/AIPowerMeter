from distutils.core import setup
setup(name='deep_learning_power_measure',
      version='1.0',
      packages=['power_consumption_measure'],
      )
install_requires=[
       'psutil',
       'pandas',
       'numpy',
       'networkx',
   ],
