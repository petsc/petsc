#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    # build on harley
    configure_options = [
        '--with-gnu-compilers=0',
        '--with-mpi-include=/home/petsc/software/mpich-1.2.2.3/alpha/include',
        '--with-mpi-lib=/home/petsc/software/mpich-1.2.2.3/alpha/lib/libmpich.a',
        '--with-mpirun=mpirun',
        '-PETSC_ARCH=osf5.0',
        '-PETSC_DIR=/sandbox/petsc/petsc-test',
        '--with-blas=/home/petsc/software/fblaslapack/alpha/libfblas.a',
        '--with-lapack=/home/petsc/software/fblaslapack/alpha/libflapack.a'
        ]

    configure.petsc_configure(configure_options)

