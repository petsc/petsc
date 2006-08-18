#!/usr/bin/env python

# Force parmetis to not build sharedlibs on IA64 by using --with-pic=0
# [as it doesn't explicitly check if mpich libraries are built with
# -fPIC flag

#  **** --with-pic=0 does not work anymore ***
#  so mpich is built with -fPIC flag - so that parmetis gets built.

configure_options = [
  '--with-vendor-compilers=intel',
  '--with-mpi-dir=/home/balay/soft/mpich2-1.0.4-intel', 
  '--download-parmetis=1',
  '--with-pic=0'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)
