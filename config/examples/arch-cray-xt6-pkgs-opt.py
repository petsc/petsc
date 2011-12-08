#!/usr/bin/python

# Do the following before running this configure script [hopp2.nersc.gov]
#
# setenv XTPE_INFO_MESSAGE_OFF yes
# module add acml

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=cc',
    '--with-cxx=CC',
    '--with-fc=ftn',

    '--with-clib-autodetect=0',
    '--with-cxxlib-autodetect=0',
    '--with-fortranlib-autodetect=0',

    '--with-debugging=0',
    '--COPTFLAGS=-fast -mp',
    '--CXXOPTFLAGS=-fast -mp',
    '--FOPTFLAGS=-fast -mp',

    '--with-blas-lapack-lib=-L/opt/acml/4.4.0/pgi64/lib -lacml -lacml_mv',
    #'--with-mpiexec=/bin/false',
    '--with-shared-libraries=0',
    '--with-x=0',

    '--download-blacs',
    '--download-hypre',
    '--download-mumps',
    '--download-cmake=1',
    '--download-metis=1',
    '--download-parmetis',
    '--download-scalapack',
    '--download-superlu_dist',
  ]
  configure.petsc_configure(configure_options)
