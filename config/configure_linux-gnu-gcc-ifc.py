#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    # build on harley
    configure_options = [
        '--with-cc=gcc',
	'--with-fc=ifc',
	'--with-cxx=g++',
        '-PETSC_ARCH=linux-gnu-gcc-ifc',
        '-PETSC_DIR=/sandbox/petsc/petsc-test',
	'--with-blas=/home/petsc/soft/linux-rh73-intel/fblaslapack/libfblas.a',
	'--with-lapack=/home/petsc/soft/linux-rh73-intel/fblaslapack/libflapack.a'
        ]

    configure.petsc_configure(configure_options)
