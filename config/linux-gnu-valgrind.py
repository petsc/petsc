#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    configure_options = [
        '--with-mpi-dir=/home/petsc/soft/linux-rh73-mpich2/mpich2-snap-20040517',
        '--with-mpirun=mpiexec.valgrind'
        ]

    configure.petsc_configure(configure_options)
