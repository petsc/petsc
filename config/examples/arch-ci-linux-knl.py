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
    '--with-debugging=0',
    '--with-cc=mpiicc',
    '--with-cxx=mpiicpc',
    '--with-fc=mpiifort',
    '--with-memalign=64',
    '--with-memkind-dir=/nfs/gce/projects/petsc/soft/u22.04/memkind-1.14.0',
    '--with-mpiexec=mpiexec.hydra',
    # Note: Use -mP2OPT_hpo_vec_remainder=F for intel compilers < version 18.
    'COPTFLAGS=-g -O3', # -xMIC-AVX512
    'CXXOPTFLAGS=-g -O3', # -xMIC-AVX512
    'FOPTFLAGS=-g -O3', # -xMIC-AVX512
    '--with-avx512-kernels=1',
    '--with-blaslapack-dir='+os.environ['MKLROOT'],
    '--download-metis=1',
    '--download-parmetis=1',
    '--download-superlu_dist=1'
  ]
  configure.petsc_configure(configure_options)
