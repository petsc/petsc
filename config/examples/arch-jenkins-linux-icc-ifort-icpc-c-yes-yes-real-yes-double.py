#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=/home/petsc/soft/linux-Ubuntu_12.04-x86/mpich-3.1.3-intel/bin/mpicc',
    '--with-fc=/home/petsc/soft/linux-Ubuntu_12.04-x86/mpich-3.1.3-intel/bin/mpif90',
    '--with-cxx=/home/petsc/soft/linux-Ubuntu_12.04-x86/mpich-3.1.3-intel/bin/mpicxx',
    '--with-blas-lapack-dir=/soft/com/packages/intel/13/update5/mkl',
    '--with-clanguage=c',
    '--with-shared-libraries=yes',
    '--with-debugging=yes',
    '--with-scalar-type=real',
    '--with-64-bit-indices=yes',
    '--with-precision=double',
    ]
  configure.petsc_configure(configure_options)
