#!/usr/bin/env python
#
# use 'scmpicc --pathscale' etc on cross-compile node
configure_options = [
  '--with-cc=mpicc',
  '--with-cxx=mpicxx',
  '--with-fc=mpif90',
  '--LIBS=-lpathfstart -lpathfortran -lpathfstart',
  '--with-fortranlib-autodetect=0',
  #
  # For link errors of type: "relocation truncated to fit: R_MIPS_GPREL16"
  # -G0 option would be needed, along with fblaslapack
  # '--download-f-blas-lapack=1',
  #
  # Recommend: -Ofast -IPA -ffast-math -CG:locs_best=1
  '--with-debugging=0',
  'COPTFLAGS=-O3 -ffast-math',
  'FOPTFLAGS=-O3 -ffast-math',
  'CXXOPTFLAGS=-O3 -ffast-math',

  '--with-batch=1',
  '--known-mpi-shared=0',

  '--known-memcmp-ok',
  '--known-sizeof-char=1',
  '--known-sizeof-void-p=8',
  '--known-sizeof-short=2',
  '--known-sizeof-int=4',
  '--known-sizeof-long=8',
  '--known-sizeof-size_t=8',
  '--known-sizeof-long-long=8',
  '--known-sizeof-float=4',
  '--known-sizeof-double=8',
  '--known-bits-per-byte=8',
  '--known-sizeof-MPI_Comm=4',
  '--known-sizeof-MPI_Fint=4',
  '--known-mpi-long-double=1',

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

  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
    
