#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-blas-lapack-lib=-L/home/petsc/soft/f2cblaslapack libf2clapack.lib libf2cblas.lib',
    '--with-cc=win32fe cl',
    '--with-cxx=0',
    '--with-fc=0',
    '--with-mpi=0',
    'PETSC_ARCH=arch-mswin-uni',
  ]
  configure.petsc_configure(configure_options)

