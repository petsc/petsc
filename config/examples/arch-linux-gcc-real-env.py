#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  os.environ['CC'] = 'gcc'
  os.environ['CFLAGS']='-I/home/sarich'
  os.environ['FC']='gfortran'
  os.environ['F77FLAGS']='-g'
  os.environ['F90FLAGS']='-g'

  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-fblaslapack=1',
    '--with-cc=gcc',
    '--with-clanguage=c',
    '--with-shared-libraries=1',
    '--with-python=1',
    '--PETSC_ARCH=arch-linux-gcc-real-env',
    ]
#    '--with-environment-variables',

  configure.petsc_configure(configure_options)
