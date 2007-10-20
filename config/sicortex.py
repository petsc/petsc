#!/usr/bin/env python
#
configure_options = [
  '--with-cc=scmpicc --pathscale',
  '--with-cxx=scmpicxx --pathscale',
  '--with-fc=scmpif90 --pathscale',
  '--LIBS=-lpathfstart -lpathfortran -lpathfstart',
  '--with-fortranlib-autodetect=0',

  # -G0 is to avoid the following error at linktime:
  # "relocation truncated to fit: R_MIPS_GPREL16"
  '--with-debugging=0',
  'COPTFLAGS=-g -O2 -G0',
  'FOPTFLAGS=-g -O2 -G0',
  'CXXOPTFLAGS=-g -O2 -G0',

  '--with-batch=1',
  '--with-mpi-shared=0',

  '--with-memcmp-ok',
  '--sizeof_char=1',
  '--sizeof_void_p=8',
  '--sizeof_short=2',
  '--sizeof_int=4',
  '--sizeof_long=8',
  '--sizeof_size_t=8',
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
  '--download-chaco=1',
  '--download-prometheus=1',

  # PETSc configure cannot cross-compile the following packages
  #'--download-mpe=1',
  # '--download-fftw',
  #'--download-hdf5=1',
  #'--download-sundials=1',
  #'--download-hypre=1',

  # not required normally. Its used due to -G0 [all sources should be
  # compiled with it]
  '--download-f-blas-lapack=1',
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
    
