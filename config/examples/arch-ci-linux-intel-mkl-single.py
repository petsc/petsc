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
    '--with-cc=mpiicc',
    '--with-cxx=mpiicpc',
    '--with-fc=mpiifort',
    '--with-mpiexec=mpiexec.hydra',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-precision=single',
    '--with-blaslapack-dir='+os.environ['MKLROOT'],
    '--with-mkl_pardiso-dir='+os.environ['MKLROOT'],
    '--with-mkl_cpardiso-dir='+os.environ['MKLROOT'],
  ]
  configure.petsc_configure(configure_options)
