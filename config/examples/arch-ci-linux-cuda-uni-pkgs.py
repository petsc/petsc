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
    '--download-p4est=1',
    '--with-zlib=1',
    # stress-test h2opus: mpiuni and CPU code while PETSc has GPU support
    '--download-h2opus',
    # need to pin c++14 since nvcc uses gcc 8 (with std=c++14), petsc uses gcc 9 (with
    # std=c++17), and interleaving this specific compiler and version combination seems to
    # break linkage of static constexpr member variables
    '--with-cxx-dialect=c++14',
    '--with-shared-libraries=1',
  ]
  configure.petsc_configure(configure_options)

