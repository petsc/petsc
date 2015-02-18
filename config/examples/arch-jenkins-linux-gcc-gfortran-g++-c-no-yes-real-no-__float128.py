#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=/usr/bin/mpicc',
    '--with-fc=/usr/bin/mpif90',
    '--with-cxx=/usr/bin/mpicxx',
    '--with-clanguage=c',
    '--with-shared-libraries=no',
    '--with-debugging=yes',
    '--with-scalar-type=real',
    '--with-64-bit-indices=no',
    '--with-precision=__float128',
    ]
  configure.petsc_configure(configure_options)
