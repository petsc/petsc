#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    # build on schwinn
    # Note: Intel 7.1 Fortran cannot work with g++ 3.3
    configure_options = [
        '--with-cc=gcc',
	'--with-fc=ifc',
	'--with-cxx=0',
        '-PETSC_ARCH=linux-gnu-gcc-ifc',
        '-PETSC_DIR=/sandbox/petsc/petsc-test',
	'--with-blas-lapack-dir=/home/petsc/soft/linux-rh73-intel/fblaslapack',
        '--with-mpi=0'
        ]

    configure.petsc_configure(configure_options)
