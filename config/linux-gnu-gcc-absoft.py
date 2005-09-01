#!/home/petsc/soft/linux-debian_sarge/python-2.2/bin/python

configure_options = [
  '--with-cc=gcc',
  '--with-fc=f90',
  '--with-cxx=g++',
  '--with-clanguage=c++',
  '--with-blas-lapack-dir=/home/petsc/soft/linux-debian_sarge-gcc-absoft/LAPACK',
  '--download-mpich=1',
  '--download-mpich-pm=gforker',
  '--download-prometheus=1',
  '--download-parmetis=1',
  '--with-matlab=0'
  ]

if __name__ == '__main__':
    import configure
    configure.petsc_configure(configure_options)

# Extra options used for testing locally
test_options = []
