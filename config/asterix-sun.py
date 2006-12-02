#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/home/balay/soft/linux-fc-sun/mpich2-1.0.3',
  '--with-shared=1',
  '--with-debugging=0',
  'LIBS=/usr/lib/libm.a'       # workarround to configure convering '/usr/lib/libm.a' to '-lm'
  ]
  
if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
