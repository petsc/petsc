#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-debugging=0',
    '--with-cc=mpiicc',
    '--with-cxx=mpiicpc',
    '--with-fc=mpiifort',
    '--with-memalign=64',
    '--with-memkind-dir=/homes/petsc/soft/memkind-v1.5.0-75-g99463a1',
    '--with-mpiexec=mpiexec.hydra',
    # use -mP2OPT_hpo_vec_remainder=F for intel compilers < version 18. Also required by runallen_cahn with 18?
    'COPTFLAGS=-g -xMIC-AVX512 -O3 -mP2OPT_hpo_vec_remainder=F',
    'CXXOPTFLAGS=-g -xMIC-AVX512 -O3 -mP2OPT_hpo_vec_remainder=F',
    'FOPTFLAGS=-g -xMIC-AVX512 -O3 -mP2OPT_hpo_vec_remainder=F',
    '--with-blaslapack-dir='+os.environ['MKLROOT'],
    '--download-metis=1',
    '--download-parmetis=1',
    '--download-superlu_dist=1',
  ]
  configure.petsc_configure(configure_options)
