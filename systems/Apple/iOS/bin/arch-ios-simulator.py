#!/usr/bin/env python
# See iosbuilder.py for instructions
# Note that this "cheats" and runs all the ./configure tests on the Mac NOT on the iPhone
#     but this is ok because the answers are the same.
# Use a 32 bit compile so this will only work on recent 64 bit OS devices
# No need to provide BLAS/LAPACK because Mac and iOS support the same Accelerate framework
#
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--download-sowing=1',
    '--download-c2html=1',
    '--with-hwloc=0',
    '--with-mpi=0',
    '--with-ios=1',
    '--with-valgrind=0',
    '--with-opengles=1',
    '--with-x=0',
    '--with-fc=0',
    '--known-blaslapack-mangling=underscore',   # ios only supports this mangling
    'PETSC_ARCH=arch-ios-simulator',
  ]
  configure.petsc_configure(configure_options)
