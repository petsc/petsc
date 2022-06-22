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
    '--with-make-test-np=15',
    #'--with-make-test-np=4', Disabled for now - OpenMPI works with 15 so hopefully future MPICH release will fix its GPU memory usage to be lower - similar to OpenMPI
    #'--download-mpich',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-scalar-type=complex',
    '--with-precision=single',
    '--with-cuda-dir=/usr/local/cuda-11.7',
    '--with-mpi-f90module-visibility=0',
  ]

  configure.petsc_configure(configure_options)
