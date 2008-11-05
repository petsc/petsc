#!/bin/env python

configure_options = [
  '--with-64-bit-pointers=1',
  '--with-mpi-compilers=0',
  '--with-vendor-compilers=solaris',
  '--download-mpich=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
