from distutils.core import setup, Extension
from Cython.Build import cythonize

setup(ext_modules = cythonize(Extension(
           "model",
           sources=["model.pyx","np_mtwister.c"],
           language="c",
      )))