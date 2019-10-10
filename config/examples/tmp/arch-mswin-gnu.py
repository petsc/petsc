#!/usr/bin/env python

configure_options = [
  # Blas autodetec with cygwin blas at /usr/lib/liblapack,a,libblas.a
  '--with-mpi-dir=/home/petsc/soft/mpich-3.1',
  '--with-shared-libraries=0',
  '--with-debugging=0',
  # not using -g so that the binaries are smaller
  'COPTFLAGS=-O',
  'FOPTFLAGS=-O',
  'CXXOPTFLAGS=-O',
  '--with-visibility=0',
  'FFLAGS=-fno-backtrace -ffree-line-length-0',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
