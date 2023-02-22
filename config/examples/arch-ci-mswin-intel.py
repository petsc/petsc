#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--download-fblaslapack=1',
    '--with-cc=icl',
    '--with-cxx=icl',
    '--with-fc=ifort',
    '--with-mpi-include=[/cygdrive/c/PROGRA~2/MICROS~2/MPI/Include/,/cygdrive/c/PROGRA~2/MICROS~2/MPI/Include/x64]',
    '--with-mpi-lib=[/cygdrive/c/PROGRA~2/MICROS~2/MPI/lib/x64/msmpifec.lib,/cygdrive/c/PROGRA~2/MICROS~2/MPI/lib/x64/msmpi.lib]',
    '--with-mpiexec=/cygdrive/c/PROGRA~1/MICROS~2/Bin/mpiexec',
    '--with-shared-libraries=0',
    '--with-strict-petscerrorcode',
  ]
  configure.petsc_configure(configure_options)
