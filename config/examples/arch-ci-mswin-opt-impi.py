#!/usr/bin/python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')
if not os.path.isdir(petsc_hash_pkgs): os.mkdir(petsc_hash_pkgs)

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-debugging=0',
    '--with-blaslapack-lib=-L/cygdrive/c/PROGRA~2/INTELS~1/COMPIL~2/windows/mkl/lib/intel64 mkl_intel_lp64_dll.lib mkl_sequential_dll.lib mkl_core_dll.lib',
    '--with-cc=win32fe cl',
    '--with-cxx=win32fe cl',
    '--with-fc=win32fe ifort',
    '--with-mpi-include=/cygdrive/c/PROGRA~2/INTELS~1/mpi/20172~1.187/intel64/include',
    '--with-mpi-lib=/cygdrive/c/PROGRA~2/INTELS~1/mpi/20172~1.187/intel64/lib/impi.lib',
    '--with-mpiexec=/cygdrive/c/PROGRA~2/INTELS~1/mpi/20172~1.187/intel64/bin/mpiexec -localonly',
    '--with-shared-libraries=0',
    'DATAFILESPATH=c:/cygwin64/home/petsc/datafiles',
  ]
  configure.petsc_configure(configure_options)
