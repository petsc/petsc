#!/usr/bin/env python3
#
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  pkgsdir = '/home/petsc/soft/master-gcc-pkgs-opt'
  configure_options = [
    '--with-clanguage=c',
    '--with-shared-libraries=yes',
    '--with-debugging=no',
    'DATAFILESPATH=/home/petsc/datafiles',
    '--with-sowing-dir=/home/petsc/soft/sowing-v1.1.25-p1',
    '--with-mpi-dir='+pkgsdir,
    '--with-metis-dir='+pkgsdir,
    '--with-parmetis-dir='+pkgsdir,
    '--with-scalapack-dir='+pkgsdir,
    '--with-mumps-dir='+pkgsdir,
    '--with-zlib-dir='+pkgsdir,
    '--with-hdf5-dir='+pkgsdir,
    '--with-netcdf-dir='+pkgsdir,
    '--with-pnetcdf-dir='+pkgsdir,
    '--with-exodusii-dir='+pkgsdir,
    '--with-ml-dir='+pkgsdir,
    '--with-suitesparse-dir='+pkgsdir,
    '--with-triangle-dir='+pkgsdir,
    '--with-chaco-dir='+pkgsdir,
    '--with-ctetgen-dir='+pkgsdir,
    ]
  configure.petsc_configure(configure_options)
