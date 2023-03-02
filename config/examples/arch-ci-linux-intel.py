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
    'COPTFLAGS=-g -O -fp-model=precise',
    'FOPTFLAGS=-g -O -fp-model=precise',
    'CXXOPTFLAGS=-g -O -fp-model=precise',
    '--with-blaslapack-dir='+os.environ['MKLROOT'],
    '--with-mkl_pardiso-dir='+os.environ['MKLROOT'],
    '--download-mpich=1',
    '--download-triangle=1',
    '--download-ctetgen=1',
    '--download-tetgen=1',
    '--download-p4est=1',
    '--download-zlib=1',
    '--download-codipack=1',
    '--download-adblaslapack=1',
    '--download-kokkos',
    '--download-kokkos-cmake-arguments=-DKokkos_ENABLE_DEPRECATION_WARNINGS=OFF', # avoid warnings caused by broken [[deprecated]] in Intel compiler
    '--download-cmake', # need cmake-3.16+ to build Kokkos
    '--download-raja',
    '--with-strict-petscerrorcode',
  ]
  configure.petsc_configure(configure_options)
