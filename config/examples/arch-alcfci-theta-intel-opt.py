#!/usr/bin/python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  #'--package-prefix-hash='+petsc_hash_pkgs,
  '--with-make-np=8',
  '--with-cc=cc',
  '--with-cxx=CC',
  '--with-fc=ftn',
  '--with-debugging=0',
  '--COPTFLAGS=-g -xMIC-AVX512 -O3',
  '--CXXOPTFLAGS=-g -xMIC-AVX512 -O3',
  '--FOPTFLAGS=-g -xMIC-AVX512 -O3',
  '--LDFLAGS=-dynamic',
  '--LIBS=-lstdc++',
  '--with-blaslapack-lib=-mkl -L/opt/intel/compilers_and_libraries_2018.0.128/linux/mkl/lib/intel64',
  '--with-mkl_sparse=0',
  '--with-mkl_sparse_optimize=0',
  '--download-chaco=1',
  '--download-exodusii=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--with-hdf5=1',
  '--with-netcdf-dir=/opt/cray/pe/netcdf/4.6.3.1/INTEL/18.0/',
  '--with-pnetcdf-dir=/opt/cray/pe/parallel-netcdf/1.11.1.0/INTEL/18.0',
  '--with-zlib=1',
  '--with-batch=1',
  '--known-mpi-long-double=1',
  '--known-mpi-int64_t=1',
  '--known-mpi-c-double-complex=1',
  '--known-64-bit-blas-indices=0',
]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
