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
  '--with-cc=mpicc.ibm-8.0',
  '--with-cxx=mpicxx.ibm-8.0',
  '--with-fc=mpif77.ibm-10.1 -qnosave',
  '--with-mpi-dir=/bgl/BlueLight/ppcfloor/bglsys',  # required by BLACS to get mpif.h


  '--with-is-color-value-type=short',
  '--with-shared=0',
  
  #'-COPTFLAGS=-O3 -qbgl -qarch=440 -qtune=440',
  #'-CXXOPTFLAGS=-O3 -qbgl -qarch=440 -qtune=440',
  #'-FOPTFLAGS=-O3 -qbgl -qarch=440 -qtune=440',
  #'--with-debugging=0',

  '--with-fortran-kernels=1'

  '--with-batch=1',
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

  '--download-f-blas-lapack=1',
  '--download-hypre=1',
  '--download-spooles=1',
  '--download-superlu=1',
  '--download-parmetis=1',
  '--download-superlu_dist=1',
  '--download-blacs=1',
  '--download-scalapack=1',
  '--download-mumps=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
    
