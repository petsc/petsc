#!/usr/bin/python3
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  mpilibdir = os.path.join(os.environ['NMPI_ROOT'],'lib64','ve')
  configure_options = [
    # NEC MPI wrappers (as of version 2.15.0) explicitly list libmpi.a when linking and not -lmpi
    # our checkSharedLinker configura test fails and PETSc will build static libraries
    # uncomment the next two lines if you need PETSc as a shared library
    # '--LDFLAGS=-Wl,-rpath,' + mpilibdir + '-L' + mpilibdir + ' -lmpi',
    # '--with-shared-libraries=1',
    '--with-debugging=0',
    # Need CXX support, and my default installation does not have system g++
    '--download-sowing-configure-arguments=CC=ncc CXX=nc++',
    'PETSC_ARCH=arch-necve',
  ]
  configure.petsc_configure(configure_options)
