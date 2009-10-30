#!/usr/bin/env python

configure_options = [
  # Blas autodetec with cygwin blas at /usr/lib/liblapack,a,libblas.a
  # MPICH2 binary install autodtect in c:/Program Files/MPICH2
  '--with-cc=gcc',
  '--with-fc=g77',
  '--with-cxx=g++',
  'DATAFILESPATH=/home/sbalay/datafiles',
  '--with-mpiexec=mpiexec --localonly',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
