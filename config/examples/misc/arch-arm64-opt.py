#!/usr/bin/python
#run this script like a normal PETSc configure script within PETSc Folder.
from __future__ import print_function
import os
import sys

configure_options = [
# iOS batch information
  '--with-batch=1',
  '--known-level1-dcache-size=32768',
  '--known-level1-dcache-linesize=32',
  '--known-level1-dcache-assoc=2',
  '--known-memcmp-ok=1',
  '--known-sizeof-char=1',
  '--known-sizeof-void-p=8',
  '--known-sizeof-short=2',
  '--known-sizeof-int=4',
  '--known-sizeof-long=8',
  '--known-sizeof-long-long=8',
  '--known-sizeof-float=4',
  '--known-sizeof-double=8',
  '--known-sizeof-size_t=8',
  '--known-bits-per-byte=8',
  '--known-snrm2-returns-double=0',
  '--known-sdot-returns-double=0',
  '--known-64-bit-blas-indices=0',

# iOS doesn't support fortran or mpi
  '--with-fc=0',
  '--with-mpi=0',

# Blas Lapack Libraries
  '--download-f2cblaslapack=1',
  # when allready available use this
  #'--with-lapack-lib=ENTER THE PATH HERE/libf2clapack.a',
  #'--with-blas-lib=ENTER THE PATH HERE/libf2cblas.a',
  # with PETSc 3.4 and newer this is possible for iOS, to run just import Accelerate Framework within Xcode:
  #'--known-blaslapack-mangling=underscore',

# this is for ios. you can change the sdk if needed to a newer/older version. Change arch, if you want to build for other architecture. ((A4,A5)-armv7,A6-armv7s,(A7,A8,A9,A10)-arm64,for simulator just ignore this line and add parameter for Simulator. For inspiration also see Onelab (open-sourche ios-app http://onelab.info/)
  '--CC=/usr/bin/llvm-gcc -arch arm64 -miphoneos-version-min=9.0 -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS9.3.sdk',


# Run with zero, for debugging with 1
  '--with-debugging=0',
# Provides Installation directory
  '--PETSC_ARCH=arch-arm64-opt',

]
if __name__ == '__main__':
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  print(configure_options)
  print(configure.petsc_configure(configure_options))
