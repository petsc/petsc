#!/usr/bin/env python

if __name__ == '__main__':
  import configure
  
  configure_options = [
    '--with-mpi-dir=/software/mpich-1.2.5.2'
    ]

  configure.petsc_configure(configure_options)
