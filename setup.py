from distutils.core import setup
import setuptools
setup(name='deep_learning_power_measure',
      version='1.0',
      packages=setuptools.find_packages(include='deep_learning_power_measure*'),
      )
install_requires=[
       'psutil',
       'pandas',
       'numpy',
       'networkx',
   ],
