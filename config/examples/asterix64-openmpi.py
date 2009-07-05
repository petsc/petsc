#!/usr/bin/env python

configure_options = [
  #OpenMPI provides broken compilers and expects users to set LD_LIBRARY_PATH - before invoking them
  'LIBS=-Wl,-rpath,/home/balay/soft/linux64/openmpi-1.3.2/lib',
  '--with-mpi-dir=/home/balay/soft/linux64/openmpi-1.3.2',
  '--with-clanguage=cxx',
  '--with-debugging=0',
  '--with-log=0',
  '--with-shared=1'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
