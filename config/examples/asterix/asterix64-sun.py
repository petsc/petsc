#!/usr/bin/env python

configure_options = [
  '--with-mpi-dir=/home/balay/soft/linux64/mpich2-1.1-sun',
  '--with-shared-libraries=1',
  '--with-debugging=0',
  #'CFLAGS=-g',                 # workarround for optimzier bug with gltr.c
  #'LIBS=/usr/lib/libm.a'       # workarround to configure convering '/usr/lib/libm.a' to '-lm'
  ]
  
if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
