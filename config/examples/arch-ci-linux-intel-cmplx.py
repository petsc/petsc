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
    'CC=icc',
    'CXX=icpc',
    'FC=ifort',
    'COPTFLAGS=-g -O',
    'FOPTFLAGS=-g -O',
    'CXXOPTFLAGS=-g -O',
    '--with-scalar-type=complex',
    '--with-blaslapack-dir='+os.environ['MKLROOT'],
    '--with-mkl_pardiso-dir='+os.environ['MKLROOT'],
    '--download-mpich',
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
    '--download-petsc4py',
    '--download-slepc',
    '--download-slepc-configure-arguments="--download-slepc4py"',
    '--download-scalapack',
    '--download-strumpack',
    '--download-suitesparse',
    '--download-superlu',
    '--download-superlu_dist',
    '--download-tetgen',
    '--download-triangle',
    '--download-zlib',
  ]
  configure.petsc_configure(configure_options)
