#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')
if not os.path.isdir(petsc_hash_pkgs): os.mkdir(petsc_hash_pkgs)

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--with-clanguage=c',
  '--with-shared-libraries=yes',
  '--with-debugging=no',
  '--download-mpich',
  '--download-metis',
  '--download-parmetis',
  '--download-scalapack',
  '--download-mumps',
  '--download-zlib',
  '--download-hdf5',
  '--download-netcdf',
  '--download-pnetcdf',
  '--download-exodusii',
  '--download-ml',
  '--download-suitesparse',
  '--download-triangle',
  '--download-chaco',
  '--download-ctetgen',
]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
