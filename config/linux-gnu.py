#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    configure_options = [
        '--with-mpi-include=/home/petsc/soft/linux-rh73/mpich-1.2.4/include',
        '--with-mpi-lib=[/home/petsc/soft/linux-rh73/mpich-1.2.4/lib/libmpich.a,/home/petsc/soft/linux-rh73/mpich-1.2.4/lib/libpmpich.a]',
        '--with-mpirun=mpirun'
        ]

    configure.petsc_configure(configure_options)
