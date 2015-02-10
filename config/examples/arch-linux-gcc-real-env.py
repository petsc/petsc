#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  os.environ['CC'] = 'gcc'
  #os.environ['FC'] = 'gfortran'

  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-mpi=0',
    '--with-clanguage=c',
    '--enable-shared',
    '--enable-fortran=0',
    '--with-environment-variables',
    '--PETSC_ARCH=arch-linux-gcc-real-env'
    ]


  configure.petsc_configure(configure_options)
