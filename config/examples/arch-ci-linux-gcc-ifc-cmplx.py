#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

# find the ifort libs location
from shutil import which
ifort_lib_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(which('ifort')))),'compiler','lib','intel64')
mpich_install_dir='/nfs/gce/projects/petsc/soft/u22.04/mpich-4.1-gcc1130-ifort20221019'
mpich_lib_dir=os.path.join(mpich_install_dir,'lib')

configure_options = [
  '--package-prefix-hash='+petsc_hash_pkgs,
  # cannot use download-mpich with fortranlib-autodetect=0 so disabling
  #'--with-cc=gcc',
  #'--with-fc=ifort',
  #'--with-cxx=g++',
  #'--download-mpich=1',
  #'--download-mpich-pm=gforker',
  '--with-mpi-dir='+mpich_install_dir,
  'LIBS=-L'+ifort_lib_dir+' -lifport -lifcoremt_pic -limf -lsvml -lm -lipgo -lirc -lpthread -L'+mpich_lib_dir+' -lmpifort -lmpi -lgcov',

  'COPTFLAGS=-g -O',
  'FOPTFLAGS=-g -O',
  'CXXOPTFLAGS=-g -O',

  '--with-scalar-type=complex',
  '--download-hdf5',
  '--with-zlib=1',
  '--download-kokkos=1',
  '--download-kokkos-kernels=1',
  '--download-fblaslapack=1',
  '--with-strict-petscerrorcode',
  '--with-coverage',
  ]

if __name__ == '__main__':
  import sys,os
  sys.path.insert(0,os.path.abspath('config'))
  import configure
  configure.petsc_configure(configure_options)
