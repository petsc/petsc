#!/usr/bin/env python

# This version requires modified mpicc/mpif77 scripts - that
# a) work with IBM cross compilers - i.e use:
# CC=" /soft/apps/ibmcmp-20050414/vac/7.0/bin/xlc -F /etc/opt/ibmcmp/vac/7.0/blrts-vac.cfg"

# b) do not give unresolved symbols on compiling mpi codes. The
# current fix is to link with some additional libraries: 
# "-lc -lnss_files -lnss_dns -lresolv"


configure_options = [
  '--with-cc=/home/balay/bin-new/mpicc.ibm',
  '--with-fc=/home/balay/bin-new/mpif77.ibm',
  '--with-blas-lapack-dir=/home/balay/software/fblaslapack/ibm-O3',
  '--with-shared=0',
  
  '-COPTFLAGS=-qbgl -qarch=440d -qtune=440 -O3',
  '-FOPTFLAGS=-qbgl -qarch=440d -qtune=440 -O3',
  
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
    
