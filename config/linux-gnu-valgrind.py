#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    configure_options = [
        '--with-mpi-dir=/home/petsc/soft/linux-rh73-mpich2/mpich2--0.97',
        '--with-mpirun=/sandbox/petsc/petsc-dev/bin/mpiexec.valgrind',
        '--with-cxx=g++',
        '--with-matlab=0'
        ]

    configure.petsc_configure(configure_options)
