#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    # build on harley
    configure_options = [
        '--with-cc=icc',
	'--with-fc=ifc -w90 -w',
	'--with-cxx=icc -Kc++ -Kc++eh',
        '--with-mpi-include=/home/petsc/soft/linux-rh73-intel/mpich-1.2.5/include',
        '--with-mpi-lib=[/home/petsc/soft/linux-rh73-intel/mpich-1.2.5/lib/libmpich.a,libpmpich.a]',
        '--with-mpirun=mpirun',
        '-PETSC_ARCH=linux-gnu-intel',
        '-PETSC_DIR=/sandbox/petsc/petsc-test',
	'--with-blas=/home/petsc/soft/linux-rh73-intel/fblaslapack/libfblas.a',
	'--with-lapack=/home/petsc/soft/linux-rh73-intel/fblaslapack/libflapack.a'
        ]

    configure.petsc_configure(configure_options)
