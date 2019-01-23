#!/usr/bin/env python3
#
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-clanguage=c',
    '--with-shared-libraries=yes',
    '--with-debugging=no',
    'DATAFILESPATH=/home/petsc/datafiles',
    '--with-mpi-dir=/home/petsc/soft/mpich-3.3b1',
    '--with-sowing-dir=/home/petsc/soft/sowing-v1.1.25-p1',
    '--with-metis-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-parmetis-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-scalapack-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-mumps-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-zlib-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-hdf5-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-netcdf-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-pnetcdf-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-exodusii-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-ml-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-suitesparse-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-triangle-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-chaco-dir=/home/petsc/soft/gcc-opt-pkgs',
    '--with-ctetgen-dir=/home/petsc/soft/gcc-opt-pkgs',
    ]
  configure.petsc_configure(configure_options)
