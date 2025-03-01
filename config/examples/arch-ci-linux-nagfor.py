#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  '--download-fblaslapack',
  '--download-mpich=https://web.cels.anl.gov/projects/petsc/download/externalpackages/mpich-4.2.3.tar.gz',
  '--with-fc=petscnagfor',
  '--with-strict-petscerrorcode',
  '--with-coverage',
]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
