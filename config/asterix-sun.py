#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/home/balay/soft/linux-fc-sun/mpich2-1.0.3',
  '--with-shared=1',
  '--with-debugging=0',
  '--with-vendor-compilers=solaris'
  ]
  
if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
