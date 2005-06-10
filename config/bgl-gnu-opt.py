#!/usr/bin/env python

# This version requires modified mpicc/mpif77 scripts - that do
# not give unresolved symbols on compiling mpi codes. The current
# fix is to link with some additional libraries:
# "-lc -lnss_files -lnss_dns -lresolv"
#
# Also the default fortran namemangling chnaged - so the usage
# of iarg_()/getargc_() internal compiler symbols does not work
# without a minor hack to zstart.c sourcefile
#
configure_options = [
  '--with-cc=/home/balay/bin/mpicc.gnu',
  '--with-fc=/home/balay/bin/mpif77.gnu',
  '--with-blas-lapack-dir=/home/balay/software/fblaslapack/gnu-O3',
  '--with-shared=0',
  
  '-COPTFLAGS=-O3',
  '-FOPTFLAGS=-O3',
  
  '--can-execute=0',
  '--sizeof_void_p=4',
  '--sizeof_short=2',
  '--sizeof_int=4',
  '--sizeof_long=4',
  '--sizeof_long_long=8',
  '--sizeof_float=4',
  '--sizeof_double=8',
  '--bits_per_byte=8',
  '--sizeof_MPI_Comm=4',
  '--sizeof_MPI_Fint=4'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
    
