#!/usr/bin/env python
  
configure_options = [
  '--with-mpi-dir=/home/petsc/soft/mpich2-1.0.2p1',
  '--with-blocksolve95=1',
  '--with-blocksolve95-include=/home/petsc/soft/BlockSolve95',
  '--with-blocksolve95-lib=/home/petsc/soft/BlockSolve95/lib/libO/freebsd/libBS95.a',
  '--with-shared=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
