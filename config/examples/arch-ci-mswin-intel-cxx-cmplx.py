#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

mpiexecf='"/cygdrive/c/Program Files/Microsoft MPI/Bin/mpiexec"'
mpidirf='"/cygdrive/c/Program Files (x86)/Microsoft SDKs/MPI"'
mpiexec=os.popen('cygpath -u '+os.popen('cygpath -ms '+mpiexecf).read()).read().strip()
mpidir=os.popen('cygpath -u '+os.popen('cygpath -ms '+mpidirf).read()).read().strip()

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--download-fblaslapack=1',
    '--with-cc=win32fe icl',
    '--with-cxx=win32fe icl',
    '--with-fc=win32fe ifort',
    '--with-clanguage=cxx',
    '--with-scalar-type=complex',
    '--with-mpi-include=['+mpidir+'/Include,'+mpidir+'/Include/x64]',
    '--with-mpi-lib=['+mpidir+'/lib/x64/msmpifec.lib,'+mpidir+'/lib/x64/msmpi.lib]',
    '--with-mpiexec='+mpiexec,
    '--with-shared-libraries=0',
    '--with-mpi-f90module-visibility=0',
  ]
  configure.petsc_configure(configure_options)
