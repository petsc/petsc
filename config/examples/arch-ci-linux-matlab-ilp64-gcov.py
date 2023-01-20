#!/usr/bin/env python3

import os
petsc_hash_pkgs=os.path.join(os.getenv('HOME'),'petsc-hash-pkgs')

# This test is done on grind.mcs.anl.gov. It uses ILP64 MKL/BLAS packaged
# with MATLAB.

# Note: regular BLAS [with 32bit integers] conflict with
# MATLAB BLAS - hence requiring -known-64-bit-blas-indices=1

# Note: MATLAB build requires petsc shared libraries

# Some versions of Matlab [R2013a] conflicted with -lgfortan - so the following workaround worked.
# export LD_PRELOAD=/usr/lib/gcc/x86_64-linux-gnu/4.6/libgfortran.so

# find MATLAB location
import os
from shutil import which
matlab_dir=os.path.dirname(os.path.dirname(which('matlab')))

if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--package-prefix-hash='+petsc_hash_pkgs,
    '--download-mpich=1', # /usr/bin/mpicc does not resolve '__gcov_merge_add'? and gcc-4.4 gives gcov errors
    '--with-display=140.221.10.20:0.0', # for matlab example with graphics
    '--with-blaslapack-dir='+matlab_dir,
    '--with-matlab=1',
# matlab-engine is deprecated, no longer needed but still allowed
    '--with-matlab-engine=1',
    '--with-shared-libraries=1',
    '-known-64-bit-blas-indices=1',
    '--with-ssl=0',
    '--with-coverage=1',
    '--with-strict-petscerrorcode',
  ]
  configure.petsc_configure(configure_options)
