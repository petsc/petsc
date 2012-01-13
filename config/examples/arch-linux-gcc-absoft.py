#!/usr/bin/env python

#!/home/petsc/soft/linux-debian_sarge/python-2.2/bin/python
# Test python-2.2 compliance [minimal python version required by PETSc configure]

configure_options = [
  '--with-cc=gcc',
  '--with-fc=f90',
  '--with-cxx=g++',
  '--with-clanguage=c++',
  '--download-f-blas-lapack=1',
  '--download-mpich=1',
  '--download-mpich-pm=gforker',
  '--download-prometheus=1',
  '--download-cmake=1',
  '--download-metis=1',
  '--download-parmetis=1',
  '--with-matlab=0'
  ]

if __name__ == '__main__':
    import sys,os
    sys.path.insert(0,os.path.abspath('config'))
    import configure
    configure.petsc_configure(configure_options)
