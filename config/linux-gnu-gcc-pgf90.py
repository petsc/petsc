#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    # Note: Intel 7.1 Fortran cannot work with g++ 3.3
    configure_options = [
        '--with-cc=gcc',
	'--with-fc=pgf90',
	'--with-cxx=g++',
	'--with-blas-lapack-dir=/home/petsc/soft/linux-rh73-intel/fblaslapack',
        '--with-mpi-dir=/home/petsc/soft/linux-rh73-gcc-pgf90/mpich-1.2.5'
        ]

    configure.petsc_configure(configure_options)
