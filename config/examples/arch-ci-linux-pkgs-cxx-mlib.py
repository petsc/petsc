#!/usr/bin/env python
import os

petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',
  '--with-single-library=0',
  '--with-clanguage=cxx',
  '--download-mpich=1',
  '--download-fblaslapack=1',
  '--download-hypre=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-ptscotch=1',
  '--download-suitesparse=1',
  '--download-triangle=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-parms=1',
  '--download-ctetgen=1',
  '--download-elemental=1',
  '--download-spai=1',
  '--download-chaco=1',
  '--download-netcdf=1',
  '--download-hdf5=1',
  '--download-adios=1',
  '--with-zlib=1',
  '--download-szlib=1',
  '--download-zstd=1',
  '--download-moab=1',
  '--with-petsc4py=1',
  '--download-mpi4py=1',
  '--download-saws',
  '--download-adolc',
  '--download-colpack',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
