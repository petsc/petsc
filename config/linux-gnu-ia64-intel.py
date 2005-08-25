#!/usr/bin/env python

configure_options = [
  '--with-gnu-compilers=0',
  '--with-mpi-dir=/home/balay/soft/mpich2-1.0.2p1-intel',
  '--download-parmetis=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
