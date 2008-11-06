#!/usr/bin/env python

configure_options = [
  '--with-cc=icc',
  '--with-fc=ifort',
  '--with-cxx=icpc',
  '--with-mpi-include=/home/petsc/soft/linux-rh73-intel/mpich-1.2.5.2/include',
  '--with-mpi-lib=[/home/petsc/soft/linux-rh73-intel/mpich-1.2.5.2/lib/libmpich.a,libpmpich.a]',
  '--with-mpiexec=/home/petsc/soft/linux-rh73-intel/mpich-1.2.5.2/bin/mpiexec',
  '--with-blas-lapack-dir=/home/petsc/soft/linux-rh73-intel/mkl-52'
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
