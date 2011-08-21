#!/usr/bin/env python

configure_options = [
  # path set to $PETSC_DIR/bin/win32fe
  '--with-vendor-compilers=borland',
  '--with-fc=0',
  '--with-ranlib=true',
  '--with-blas-lapack-dir=/home/sbalay/soft/borland-f2cblas',
  '--download-f2cblaslapack=1',
  '--with-mpi=0',
  'DATAFILESPATH=/home/sbalay/datafiles',
  ]
  
if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
