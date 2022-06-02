#!/usr/bin/python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

oadirf='"/cygdrive/c/Program Files (x86)/Intel/oneAPI"'
oadir=os.popen('cygpath -u '+os.popen('cygpath -ms '+oadirf).read()).read().strip()
oamkldir=oadir+'/mkl/2022.1.0/lib/intel64'
oampidir=oadir+'/mpi/2021.6.0'

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-debugging=0',
    '--with-blaslapack-lib=-L'+oamkldir+' mkl_intel_lp64_dll.lib mkl_sequential_dll.lib mkl_core_dll.lib',
    '--with-cc=win32fe cl',
    '--with-cxx=win32fe cl',
    '--with-fc=win32fe ifort',
    '--with-mpi-include='+oampidir+'/include',
    '--with-mpi-lib='+oampidir+'/lib/release/impi.lib',
    '-with-mpiexec='+oampidir+'/bin/mpiexec -localonly',
  ]
  configure.petsc_configure(configure_options)
