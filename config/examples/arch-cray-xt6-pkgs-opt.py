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

    '--with-debugging=0',
    '--COPTFLAGS=-fast -mp',
    '--CXXOPTFLAGS=-fast -mp',
    '--FOPTFLAGS=-fast -mp',

    '--with-blaslapack-lib=-L/opt/acml/4.4.0/pgi64/lib -lacml -lacml_mv',
    #'--with-mpiexec=/bin/false',
    '--with-shared-libraries=0',
    '--with-x=0',

    '--download-hypre',
    '--download-mumps',
    # no with-cxx-dialect=C++11 support???? '--download-elemental=1',
    '--download-cmake=1',
    '--download-metis=1',
    '--download-parmetis',
    '--download-scalapack',
    '--download-superlu_dist',
  ]
  configure.petsc_configure(configure_options)
