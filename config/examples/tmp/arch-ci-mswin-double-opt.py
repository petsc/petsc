#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')
if not os.path.isdir(petsc_hash_pkgs): os.mkdir(petsc_hash_pkgs)

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-debugging=0',
  '--download-f2cblaslapack',
  '--with-cc=win32fe cl',
  '--with-cxx=win32fe cl',
  '--with-fc=0',
  '--with-mpi-include=[/cygdrive/c/PROGRA~2/MICROS~2/MPI/Include/,/cygdrive/c/PROGRA~2/MICROS~2/MPI/Include/x64]',
  '--with-mpi-lib=[/cygdrive/c/PROGRA~2/MICROS~2/MPI/lib/x64/msmpifec.lib,/cygdrive/c/PROGRA~2/MICROS~2/MPI/lib/x64/msmpi.lib]',
  '--with-mpiexec=/cygdrive/c/PROGRA~1/MICROS~2/Bin/mpiexec',
  '--with-shared-libraries=0',
  'DATAFILESPATH=c:/cygwin64/home/petsc/datafiles',
]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
