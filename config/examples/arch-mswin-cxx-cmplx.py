#!/usr/bin/env python

configure_options = [
  # Autodetect MPICH & Intel MKL
  # path set to $PETSC_DIR/bin/win32fe
  # Note: comment out MPI_DISPLACEMENT_CURRENT in mpif.h for MPICH2 to work with Compaq F90
  '--with-cc=win32fe cl',
  '--with-fc=win32fe f90',
  '--with-cxx=win32fe cl',
  '--with-clanguage=cxx',
  '--with-scalar-type=complex',
  'CXXFLAGS=-DMPICH_SKIP_MPICXX -MT -GR -EHsc',
  '--with-mpiexec=mpiexec --localonly',
  'DATAFILESPATH=/home/sbalay/datafiles',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
