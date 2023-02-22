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
    # Intel compilers enable GCC/clangs equivalent of -ffast-math *by default*. This is
    # bananas, so we make sure they use the same model as everyone else
    'COPTFLAGS=-g -O -fp-model=precise',
    'FOPTFLAGS=-g -O -fp-model=precise',
    'CXXOPTFLAGS=-g -O -fp-model=precise',
    '--with-precision=single',
    '--with-blaslapack-dir='+os.environ['MKLROOT'],
    '--with-mkl_pardiso-dir='+os.environ['MKLROOT'],
    '--with-mkl_cpardiso-dir='+os.environ['MKLROOT'],
    '--download-superlu_dist',
    '--download-metis',
    '--download-parmetis',
    '--with-strict-petscerrorcode',
  ]
  configure.petsc_configure(configure_options)
