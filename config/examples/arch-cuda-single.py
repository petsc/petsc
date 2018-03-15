#!/usr/bin/python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cuda=1',
    '--with-cusp=1',
    '--with-cusp-dir=/home/balay/soft/cusplibrary-g0a21327',
    '--with-precision=single',
    '--download-openblas', # default ATLAS blas on Ubuntu 14.04 breaks runex76 in src/mat/examples/tests
    '--with-clanguage=c',
  ]
  configure.petsc_configure(configure_options)
