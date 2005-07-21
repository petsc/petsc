#!/usr/bin/env python
#
# This version requires IBM's cross compilers mpxlc/mpxlf
#
# BGL has broken 'libc' dependencies. The option 'LIBS' is used to
# workarround this problem. Another workarround is to modify
# mpxlc/mpxlf [or mpicc/mpif77] scripts and make them link with these
# additional libraries.
#
# Modifying mpicc [from using gcc to blrts_xlc] would require something like:
# CC=" /soft/apps/ibmcmp-20050414/vac/7.0/bin/xlc -F /etc/opt/ibmcmp/vac/7.0/blrts-vac.cfg"
#
configure_options = [
  '-LIBS=-lc -lnss_files -lnss_dns -lresolv',
  '--with-cc=mpxlc',
  '--with-fc=mpxlf',

  '--with-blas-lapack-dir=/home/balay/software/fblaslapack/ibm-O3',
  '--with-shared=0',
  
  '-COPTFLAGS=-qbgl -qarch=440d -qtune=440 -O3',
  '-FOPTFLAGS=-qbgl -qarch=440d -qtune=440 -O3',
  '--with-debugging=0',
  '--with-fortran-kernels-bgl=1'

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
    
