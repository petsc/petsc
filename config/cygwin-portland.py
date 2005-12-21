#!/usr/bin/env python

# ******* Currently not tested **********

configure_options = [
  '--with-vendor-compilers=portland',
  # C++ compiler does not appear to be distributed with the trial version, maybe there is one?
  '--with-cxx=0',
  # Use MPIUni
  '--with-mpi=0'
  # To use MPICH-NT.1.2.5, a couple simple edits of the MPICH-NT.1.2.5 header files are necessary, then
  #'--with-mpi-dir=F:/MPI/mpich-nt.1.2.5' # Note you need to use <Drive>:/ notation for this compiler
  #
  # Autodetect PGI BLAS/LAPACK
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
