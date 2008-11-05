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
# ftp://ftp.mcs.anl.gov/pub/petsc/tmp/petsc-bgl-tools.tar.gz 
#
configure_options = [
  '--with-cc=mpicc.ibm-8.0',
  '--with-cxx=mpicxx.ibm-8.0',
  '--with-fc=mpif77.ibm-10.1 -qnosave',
  '--with-mpi-dir=/bgl/BlueLight/ppcfloor/bglsys',  # required by BLACS to get mpif.h
  '--with-clanguage=cxx',

  '--with-is-color-value-type=short',
  '--with-shared=0',
  
  '-COPTFLAGS=-O3 -qbgl -qarch=440d -qtune=440 -qmaxmem=-1',
  '-CXXOPTFLAGS=-O3 -qbgl -qarch=440d -qtune=440 -qmaxmem=-1',
  '-FOPTFLAGS=-O3 -qbgl -qarch=440d -qtune=440 -qmaxmem=-1',
  '--with-debugging=0',

  # the following option gets automatically enabled on BGL/with IBM compilers.
  # '--with-fortran-kernels=bgl'

  '--with-batch=1',
  '--with-memcmp-ok',
  '--sizeof_char=1',
  '--sizeof_void_p=4',
  '--sizeof_short=2',
  '--sizeof_int=4',
  '--sizeof_long=4',
  '--sizeof_size_t=4',
  '--sizeof_long_long=8',
  '--sizeof_float=4',
  '--sizeof_double=8',
  '--bits_per_byte=8',
  '--sizeof_MPI_Comm=4',
  '--sizeof_MPI_Fint=4',
  '--have-mpi-long-double=1',

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
    
