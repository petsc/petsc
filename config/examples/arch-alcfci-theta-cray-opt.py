#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  #'--package-prefix-hash='+petsc_hash_pkgs,
  '--with-make-np=8',
  '--with-cc=cc',
  '--with-cxx=CC',
  '--with-fc=ftn',
  #'--with-debugging=0',
  #'--COPTFLAGS=-g -O3',
  #'--CXXOPTFLAGS=-g -O3',
  #'--FOPTFLAGS=-g -O3',
  '--LDFLAGS=-dynamic',
  '--LIBS=-lstdc++',
  '--download-chaco=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--with-hdf5=1',
  '--with-netcdf-dir='+os.environ['CRAY_NETCDF_HDF5PARALLEL_PREFIX'],
  '--with-pnetcdf-dir='+os.environ['CRAY_PARALLEL_NETCDF_PREFIX'],
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
