#!/usr/bin/python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-debugging=0',
    '--with-cc=mpiicc',
    '--with-cxx=mpiicpc',
    '--with-fc=mpiifort',
    '--with-memalign=64',
    '--with-memkind-dir=/homes/petsc/soft/memkind-v1.5.0-75-g99463a1',
    '--with-mpiexec=mpiexec.hydra',
    # Note: Use -mP2OPT_hpo_vec_remainder=F for intel compilers < version 18.
    'COPTFLAGS=-g -xMIC-AVX512 -O3',
    'CXXOPTFLAGS=-g -xMIC-AVX512 -O3',
    'FOPTFLAGS=-g -xMIC-AVX512 -O3',
    '--with-avx512-kernels=1',
    '--with-blaslapack-dir='+os.environ['MKLROOT'],
    '--download-metis=1',
    '--download-parmetis=1',
    '--download-superlu_dist=1'
  ]
  configure.petsc_configure(configure_options)
