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
    'CC=icx',
    'CXX=icpx',
    'FC=ifx',
    # Intel compilers enable GCC/clangs equivalent of -ffast-math *by default*. This is
    # bananas, so we make sure they use the same model as everyone else
    'COPTFLAGS=-g -O -fp-model=precise -Wno-deprecated-non-prototype -Wno-implicit-int -Wno-implicit-function-declaration -Wno-incompatible-pointer-types -Wincompatible-function-pointer-types -Wno-unused-result',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O -fp-model=precise',
    '--with-scalar-type=complex',
    '--with-blaslapack-dir='+os.environ['MKLROOT'],
    '--with-mkl_pardiso-dir='+os.environ['MKLROOT'],
    '--download-mpich',
    '--download-bamg',
    '--download-chaco',
    '--download-codipack',
    '--download-ctetgen',
    '--download-hdf5',
    '--download-hypre',
    '--download-metis',
    '--download-mpi4py',
    # '--download-mumps',
    '--download-p4est',
    '--download-parmetis',
    '--with-petsc4py',
    '--download-slepc',
    '--download-slepc-configure-arguments="--with-slepc4py"',
    '--download-scalapack',
    '--download-strumpack',
    '--download-suitesparse',
    '--download-superlu',
    '--download-superlu_dist',
    '--download-tetgen',
    '--download-triangle',
    '--download-zlib',
    '--with-strict-petscerrorcode',
  ]
  configure.petsc_configure(configure_options)
