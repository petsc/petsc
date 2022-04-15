#!/export/home/glci/soft/python-3.7.13/bin/python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  #'--with-debugger=/bin/true',
  '--with-debugging=0',
  'FFLAGS=-ftrap=%none',

  '--with-64-bit-indices=1',
  '--with-log=0',
  '--with-info=0',
  '--with-ctable=0',
  '--with-is-color-value-type=short',
  '--with-single-library=0',

  '--with-c2html=0',
  '--with-mpi-dir=/export/home/petsc/soft/mpich-3.3.1',
  # opensolaris throws warning
  # CC: Warning: Option -std=c++03 passed to ld, if ld is invoked, ignored otherwise
  # to stderr, so don't use flag at all
  '--with-cxx-dialect=0',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
