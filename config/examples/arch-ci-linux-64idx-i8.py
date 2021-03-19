#!/usr/bin/python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-64-bit-indices',
    'FFLAGS=-Wall -ffree-line-length-0 -Wno-unused-dummy-argument -fdefault-integer-8',
    '--with-mpi-dir=/nfs/gce/projects/petsc/soft/gcc-7.4.0/mpich-3.3.2',
    '--with-mpi-f90module-visibility=0',
  ]
  configure.petsc_configure(configure_options)

