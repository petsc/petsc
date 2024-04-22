#!/usr/bin/python3
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-blaslapack-lib=-L/cygdrive/c/PROGRA~2/Intel/oneAPI/mkl/latest/lib mkl_intel_lp64_dll.lib mkl_sequential_dll.lib mkl_core_dll.lib',
    '--with-cc=icx',
    '--with-cxx=icx',
    '--with-fc=ifx',
    '--with-shared-libraries=0',
    'FPPFLAGS=-I/cygdrive/c/PROGRA~2/Intel/oneAPI/mpi/latest/include/mpi',
    '--with-mpi-include=/cygdrive/c/PROGRA~2/Intel/oneAPI/mpi/latest/include',
    '--with-mpi-lib=/cygdrive/c/PROGRA~2/Intel/oneAPI/mpi/latest/lib/impi.lib',
    '--with-mpiexec=/cygdrive/c/PROGRA~2/Intel/oneAPI/mpi/latest//bin/mpiexec -localonly',
  ]
  configure.petsc_configure(configure_options)
