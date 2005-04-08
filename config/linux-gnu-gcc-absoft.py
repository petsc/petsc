#!/home/petsc/soft/linux-debian_sarge/python-2.2/bin/python

configure_options = [
  '--with-cc=gcc',
  '--with-fc=f90',
  '--with-cxx=g++',
  '--with-language=c++',
  '--download-f-blas-lapack=1',
  '--download-mpich=1',
  '--download-mpich-pm=gforker',
  '--with-matlab=0'
  ]

if __name__ == '__main__':
    import configure
    configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
