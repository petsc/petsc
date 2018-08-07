#!/usr/bin/python
#run this script like a normal PETSc configure script within PETSc Folder.

from __future__ import print_function
import os
import sys

configure_options = [
# Android batch information
  '--with-batch=1',
  '--known-level1-dcache-size=32768',
  '--known-level1-dcache-linesize=32',
  '--known-level1-dcache-assoc=2',
  '--known-memcmp-ok=1',
  '--known-endian=little',
  '--known-sizeof-char=1',
  '--known-sizeof-void-p=4',
  '--known-sizeof-short=2',
  '--known-sizeof-int=4',
  '--known-sizeof-long=4',
  '--known-sizeof-long-long=8',
  '--known-sizeof-float=4',
  '--known-sizeof-double=8',
  '--known-sizeof-size_t=4',
  '--known-bits-per-byte=8',
  '--known-snrm2-returns-double=0',
  '--known-sdot-returns-double=0',
  '--known-64-bit-blas-indices=0',

# Android will not use mpi or Fortran on this run
# For further information on how to include Fortan standalones see:
# https://github.com/buffer51/android-gfortran
  '--with-mpi=0',
  '--with-fc=0',

# Android standalone binaries to be used by PETSc
# Put your standalone bin folder in you PATH
  '--CC=arm-linux-androideabi-gcc',
  '--CXX=arm-linux-androideabi-g++',
  '--AR=arm-linux-androideabi-gcc-ar',
  '--host=arm-linux-androideabi',

# Blas Lapack Libraries
  '--download-f2cblaslapack=1',
# when allready available use this
#'--with-lapack-lib=ENTER THE PATH HERE/libf2clapack.a',
#'--with-blas-lib=ENTER THE PATH HERE/libf2cblas.a',

  '--with-shared-libraries=0',
  '--with-debugging=0',
  '--PETSC_ARCH=arch-armv7-opt',

# These flags were requested by the system which was being tested on.
# This might not apply for other Android devices.
# In case your cross-compilation does not run, uncomment and test these flags
# '--CFLAGS=-fPIE',
# '--LDFLAGS=-fPIE -pie'
]

if __name__ == '__main__':
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  print(configure_options)
  print(configure.petsc_configure(configure_options))
