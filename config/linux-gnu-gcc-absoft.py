#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    configure_options = [
        '--with-cc=gcc',
	'--with-fc=f90',
	'--with-cxx=g++',
	'--with-blas-lib=/home/petsc/software/LAPACK/libblas_linux_absoft.a',
	'--with-lapack-lib=/home/petsc/software/LAPACK/liblapack_linux_absoft.a',
        '--with-mpi-dir=/home/petsc/software/mpich-1.2.0/linux_absoft',
        '--with-matlab=0'
        ]

    configure.petsc_configure(configure_options)
