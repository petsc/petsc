#!/usr/bin/env python

# Do the following before running this configure script [hopper.nersc.gov]
#
# setenv XTPE_INFO_MESSAGE_OFF yes
# module add acml

configure_options = [
# On cray cc,CC,ftn are eqivalent to mpicc,mpiCC,mpif90
  '--with-cc=cc',
  '--with-cxx=CC',
  '--with-fc=ftn',

  '--with-shared-libraries=0',
  '--with-debugging=0',
  '--COPTFLAGS=-fastsse -O3 -tp barcelona-64',
  '--CXXOPTFLAGS=-fastsse -O3 -tp barcelona-64',
  '--FOPTFLAGS=-fastsse -O3 -tp barcelona-64',

  '--with-clib-autodetect=0',
  '--with-cxxlib-autodetect=0',
  '--with-fortranlib-autodetect=0',

  '--with-mpiexec=/usr/common/acts/PETSc/3.0.0/bin/mpiexec.aprun',
  '--with-x=0',
  '--with-blas-lapack-lib=-L/opt/acml/4.3.0/pgi64/lib -lacml -lacml_mv',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
