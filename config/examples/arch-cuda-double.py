#!/usr/bin/python
#
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cuda=1',
    '--download-cusp=1',
    '--with-precision=double',
    '--with-clanguage=c',
  ]
  configure.petsc_configure(configure_options)
