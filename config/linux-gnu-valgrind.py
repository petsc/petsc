#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/home/petsc/soft/linux-rh73-mpich2/mpich2-0.971-CVS-200408131639',
  '--with-mpirun=/sandbox/petsc/petsc-dev/bin/mpiexec.valgrind',
  '--with-cxx=g++',
  '--with-matlab=0'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
