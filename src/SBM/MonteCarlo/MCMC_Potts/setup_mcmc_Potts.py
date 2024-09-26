import os
from setuptools import setup, Extension
import numpy as np

os.environ["CC"]="gcc"
os.environ["CXX"]="g++"
sfc_module = Extension("MonteCarlo_Potts",
                sources = ["src/SBM/MonteCarlo/MCMC_Potts/MonteCarlo_PottsMod.cpp"],
				include_dirs = [np.get_include()],
                extra_compile_args = ["-DNDEBUG", "-O3", "-std=c++17", "-fopenmp"],
                extra_link_args = ['-lgomp']
)

setup(
    name="MonteCarlo_Potts",
    version='1.0',
    description="Python Package with MonteCarlo C++ extension",
    ext_modules=[sfc_module]
)