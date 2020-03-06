#!/usr/bin/env python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  '--with-64-bit-indices=1',
  '--download-mpich=1', #openmpi gives errors of type: Error: There is no specific subroutine for the generic 'mpi_send'
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-hwloc=1',
  '--download-pastix=1',
  '--download-ptscotch=1',
  '--download-hypre=1',
  '--download-hypre-configure-arguments=--enable-bigint=no --enable-mixedint=yes', # HYPRE with mixed integers
  '--download-superlu_dist=1',
  '--donwload-suitesparse=1',
  '--download-p4est=1',
  '--with-zlib=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
