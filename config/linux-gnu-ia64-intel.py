#!/usr/bin/env python

configure_options = [
  # cannot build shared libraries are our particular test machine
  '--with-shared=0',
  '--with-gnu-compilers=0',
  '--with-mpi=0',
  '--with-blas-lib=[/usr/lib/libblas.a,/usr/lib/gcc-lib/ia64-redhat-linux/2.96/libg2c.a]',
  '--with-lapack-lib=/usr/lib/liblapack.a'
  ]

if __name__ == '__main__':
  import configure
  configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
