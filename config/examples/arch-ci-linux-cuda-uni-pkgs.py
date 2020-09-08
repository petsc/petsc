#!/usr/bin/python

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-mpi=0',
    '--with-cc=gcc',
    '--with-cxx=g++',
    '--with-fc=gfortran',
    '--with-cuda=1',
    '--download-hdf5',
    '--download-metis',
    '--download-superlu',
    '--download-mumps',
    '--with-mumps-serial',
    '--with-shared-libraries=1',
  ]
  configure.petsc_configure(configure_options)

