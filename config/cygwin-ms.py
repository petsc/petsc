#!/usr/bin/env python

configure_options = [
  # Autodetect MPICH & Intel MKL
  # path set to $PETSC_DIR/bin/win32fe
  '--with-cc=win32fe cl',
  '--with-cxx=win32fe cl',
  '--with-fc=win32fe f90',
  '--with-f90-interface=win32'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
