#!/usr/bin/env python3

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
  '--with-blaslapack-lib=-mkl -L'+os.path.join(os.environ['MKLROOT'],'lib','intel64'),
  '--with-mkl_sparse=0',
  '--with-mkl_sparse_optimize=0',
  '--download-chaco=1',
  '--download-exodusii=1',
  '--download-exodusii-cmake-arguments=-DCMAKE_C_FLAGS:STRING="-DADDC_ -fPIC -g -xMIC-AVX512 -O3"', # workaround exodusii cmake failure 'cannot automatically determine Fortran mangling'
  '--download-metis=1',
  '--download-parmetis=1',
  '--with-hdf5=1',
  '--with-netcdf-dir='+os.environ['CRAY_NETCDF_HDF5PARALLEL_PREFIX'],
  '--with-pnetcdf-dir='+os.environ['CRAY_PARALLEL_NETCDF_PREFIX'],
  '--with-zlib=1',
  '--with-batch=1',
  '--known-64-bit-blas-indices=0',
]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
