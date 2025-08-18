#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-clanguage=c',
  '--with-shared-libraries=yes',
  '--with-debugging=no',
  '--download-mpich',
  '--download-mpich-device=ch3:sock',
  '--download-metis',
  '--download-parmetis',
  '--download-scalapack',
  '--download-mumps',
  '--download-zlib',
  '--download-hdf5',
  '--download-netcdf',
  '--download-pnetcdf',
  '--download-exodusii',
  '--with-exodusii-fortran-bindings',
  '--download-ml',
  '--download-suitesparse',
  '--download-triangle',
  '--download-triangle-build-exec',
  '--download-cgns',
  #'--download-chaco', run with hdf5, exodus, but without chaco
  '--download-ctetgen',
  '--download-cmake',
  '--download-amrex',
  '--download-hypre',
  '--download-ks',
  '--download-sprng',
  '--with-ssl=1',
  '--with-tau-perfstubs=0',
  '--with-strict-petscerrorcode',
  '--with-coverage',
]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
