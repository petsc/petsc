#!/usr/bin/env python

configure_options = [
  '--with-cc=/home/petsc/soft/inux-debian_sarge-gcc4/gcc-4.1.0/bin/gcc',
  '--with-cxx=/home/petsc/soft/inux-debian_sarge-gcc4/gcc-4.1.0/bin/g++',
  '--download-mpich=1',
  '--with-sieve=1',
  '--download-boost=1',
  '--with-clanguage=cxx'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
