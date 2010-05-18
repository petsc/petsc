#!/System/Library/Frameworks/Python.framework/Versions/2.5/Resources/Python.app/Contents/MacOS/Python
if __name__ == '__main__':
  import sys
  import os
  sys.path.insert(0, os.path.abspath('config'))
  import configure
  configure_options = [
    '--with-cc=gcc -m32',
    '--with-mpi=0',
    '--with-iphone=1',
    '--download-c-blas-lapack',
    '--with-x=0',
    'PETSC_ARCH=arch-iphone',
    '--with-valgrind=0',
    '--with-fc=0',
  ]
  configure.petsc_configure(configure_options)
