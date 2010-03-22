#!/usr/bin/env python

configure_options = [
  '--with-cc=mpcc_r',
  '--with-cxx=mpcxx_r',
  '--with-fc=mpxlf_r',
  '--with-blas-lapack-lib=-lessl -L/usr/common/usg/LAPACK/3.0 -llapack -lessl',
  '--with-mpiexec=/usr/common/acts/PETSc/3.0.0/bin/mpiexec.poe',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
