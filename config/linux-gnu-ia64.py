#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    configure_options = [
        # -lg2c messes up shared libraries
         '--with-shared=0',
         '--with-mpi-dir=/home/petsc/soft/linux-ia64/mpich-1.2.5.2'
        ]

    configure.petsc_configure(configure_options)
