#!/usr/bin/env python

configure_options = [
  # Using MPICH for Windows 2000/NT available from http://www.mcs.anl.gov/mpi/mpich
  #'--with-mpi-dir=/cygdrive/c/Program\ Files/MPICH/SDK',
  #
  # Using Intel's MKL available from http://www.intel.com
  #'--with-blas-lapack-dir=/cygdrive/c/Program\ Files/Intel/MKL',
  #
  # Using Intel Compilers
  '--with-vendor-compilers=intel'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
  
