from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np
import os
import sys



os.environ["CC"]="g++"
os.environ["CXX"]="gcc"
sfc_module= Extension("C_MonteCarlo",
              sources = ["src/SBM/MonteCarlo/C_MonteCarlo.pyx"],
              include_dirs = [np.get_include()],
              #extra_compile_args = ["-O3", "-ffast-math","-march=native", "-fopenmp","-std=c++17","-I/usr/local/include" ],
              extra_compile_args = ["-O3", "-ffast-math","-march=native", "-fopenmp","-std=c++17"],
              #extra_link_args=['-lomp','-lgomp',"-fopenmp"],
              extra_link_args=['-lgomp']
) 

setup( 
  name = "C_MonteCarlo",
  cmdclass = {"build_ext": build_ext},
  ext_modules = [sfc_module]
)