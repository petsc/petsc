#!/usr/bin/python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--with-64-bit-indices',
    'FFLAGS=-Wall -ffree-line-length-0 -Wno-unused-dummy-argument -fdefault-integer-8',
    '--with-mpi-dir=/nfs/gce/projects/petsc/soft/u22.04/mpich-4.0.2',
    '--with-mpi-ftn-module=mpi_f08',
    '--with-strict-petscerrorcode',
    '--download-kokkos=1',
    '--download-kokkos-commit=1557870d70d5ac0a636d3e8873d5b4ce1bb0375b', # develop as of 5/1/2026
    '--download-kokkos-kernels=1',
    '--download-kokkos-kernels-commit=90ce916124f86173481944db6c810d67e8978bd0', # develop as of 5/1/2026
    '--with-coverage',
  ]
  configure.petsc_configure(configure_options)
