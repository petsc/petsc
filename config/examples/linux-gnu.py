#!/usr/bin/env python

configure_options = [
  '--download-mpich=1',
  '--download-mpich-pm=gforker',
  '--with-gcov=1'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
