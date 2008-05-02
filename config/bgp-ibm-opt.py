#!/usr/bin/env python
#
configure_options = [
  '--with-cc=mpixlc_r',
  '--with-cxx=mpixlcxx_r',
  '--with-fc=mpixlf77_r -qnosave',

  '--with-mpi-dir=/bgsys/drivers/ppcfloor/comm',  # required by BLACS to get mpif.h
  '--with-blas-lapack-lib=[/soft/apps/blas-lapack-lib/liblapack_bgp.a,libgoto.a]',
  '--with-x=0',

  '--with-is-color-value-type=short',
  '--with-shared=0',
  
  '-COPTFLAGS=-O3 -qarch=450d -qtune=450 -qmaxmem=-1',
  '-CXXOPTFLAGS=-O3 -qarch=450d -qtune=450 -qmaxmem=-1',
  '-FOPTFLAGS=-O3 -qarch=450d -qtune=450 -qmaxmem=-1',
  '--with-debugging=0',

  # autodetect on BGP not working?
  '--with-fortran-kernels=bgl',

  '--with-batch=1',
  '--with-mpi-shared=0',
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

  '--download-plapack=1',
  '--download-parmetis=1',
  '--download-umfpack=1',
  '--download-triangle=1',
  '--download-spooles=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-blacs=1',
  '--download-scalapack=1',
  '--download-mumps=1',
  '--download-spai=1',
  '--download-prometheus=1',
  #'--download-chaco=1', [had namespace conflict with petsc : vecscale]
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
