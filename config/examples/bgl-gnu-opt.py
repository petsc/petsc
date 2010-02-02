#!/usr/bin/env python
#
# BGL has broken 'libc' dependencies. The option 'LIBS' is used to
# workarround this problem.
#
# LIBS="-lc -lnss_files -lnss_dns -lresolv"
#
# Another workarround is to modify mpicc/mpif77 scripts and make them
# link with the corresponding compilers, and these additional
# libraries. The following tarball has the modified compiler scripts
#
# http://ftp.mcs.anl.gov/pub/petsc/tmp/petsc-bgl-tools.tar.gz 
#
configure_options = [
  '-LIBS=-lc -lc -lnss_files -lnss_dns -lresolv',
  '--with-cc=mpicc',
  '--with-cxx=mpicxx',
  '--with-fc=mpif77',

  '--download-f-blas-lapack=1',
  '--with-shared=0',
  
  '-COPTFLAGS=-O3',
  '-FOPTFLAGS=-O3',
  '--with-debugging=0',
  '--with-fortran-kernels=1',
  '--with-x=0',
  
  '--with-batch=1',
  '--known-mpi-shared=0',
  '--known-endian=big',
  '--known-memcmp-ok',
  '--known-sizeof-char=1',
  '--known-sizeof-void-p=4',
  '--known-sizeof-short=2',
  '--known-sizeof-int=4',
  '--known-sizeof-long=4',
  '--known-sizeof-size_t=4',
  '--known-sizeof-long-long=8',
  '--known-sizeof-float=4',
  '--known-sizeof-double=8',
  '--known-bits-per-byte=8',
  '--known-sizeof-MPI_Comm=4',
  '--known-sizeof-MPI_Fint=4',
  '--known-mpi-long-double=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
    
