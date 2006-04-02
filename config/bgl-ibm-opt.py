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
  '--with-cc=mpicc.ibm',
  '--with-cxx=mpicxx.ibm',
  '--with-fc=mpif77.ibm -qnosave',
  '--with-mpi-dir=/bgl/BlueLight/V1R2M1_020_2006-060110/ppc/bglsys',  # required by BLACS to get mpif.h

  '--with-is-color-value-type=short',
  '--with-blas-lapack-lib=[/opt/ibmmath/essl/4.2/lib/libesslbg.a,/soft/tools/lapack440/liblapack440.a,/soft/tools/blass440/libblas440.a]',
  '--with-shared=0',
  
  '-COPTFLAGS=-O3 -qbgl -qarch=440 -qtune=440',
  '-CXXOPTFLAGS=-O3 -qbgl -qarch=440 -qtune=440',
  '-FOPTFLAGS=-O3 -qbgl -qarch=440 -qtune=440',
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
  '--sizeof_long_long=8',
  '--sizeof_float=4',
  '--sizeof_double=8',
  '--bits_per_byte=8',
  '--sizeof_MPI_Comm=4',
  '--sizeof_MPI_Fint=4',

  '--download-hypre=1',
  '--download-spooles=1',
  '--download-superlu=1',
  '--download-superlu_dist=1',
  '--download-blacs=1',
  '--download-scalapack=1',
  '--download-mumps=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
    
