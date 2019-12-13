from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

ext1 = Extension("relaxation", ["relaxation.pyx"],
    include_dirs = [numpy.get_include()])

ext2 = Extension("gridtransfers", ["gridtransfers.pyx"],
    include_dirs = [numpy.get_include()])

setup(ext_modules=[ext1, ext2],
      cmdclass = {'build_ext': build_ext})
