#!/usr/bin/env python
#
configure_options = [
  '--with-cc=mpixlc_r',
  '--with-cxx=mpixlcxx_r',
  '--with-fc=mpixlf77_r -qnosave',

  #'--with-mpi-dir=/bgsys/drivers/ppcfloor/comm',  # required by BLACS to get mpif.h
  '--with-blas-lapack-lib=-L/soft/apps/LAPACK -llapack_bgp -L/soft/apps/LIBGOTO -lgoto',
  '--with-x=0',

  '--with-is-color-value-type=short',
  '--with-shared-libraries=0',
  
  '-COPTFLAGS=-O3 -qarch=450d -qtune=450 -qmaxmem=-1',
  '-CXXOPTFLAGS=-O3 -qarch=450d -qtune=450 -qmaxmem=-1',
  '-FOPTFLAGS=-O3 -qarch=450d -qtune=450 -qmaxmem=-1',
  '--with-debugging=0',

  # autodetect on BGP not working?
  '--with-fortran-kernels=1',

  '--with-batch=1',
  '--known-mpi-shared-libraries=0',
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
  '--known-level1-dcache-assoc=0',
  '--known-level1-dcache-linesize=32',
  '--known-level1-dcache-size=32768',

  '--download-hypre=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--download-umfpack=1',
  '--download-triangle=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-blacs=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-spai=1',
  '--download-chaco=1',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
