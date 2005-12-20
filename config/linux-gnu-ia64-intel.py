#!/usr/bin/env python

configure_options = [
  '--with-vendor-compilers=intel',
  '--with-mpi-dir=/home/balay/soft/mpich2-1.0.3-intel',
  '--download-parmetis=1',
  '--with-pic=0'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
