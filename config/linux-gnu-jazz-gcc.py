#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    configure_options = [
        '--with-mpi-dir=/soft/apps/packages/mpich-gm-1.2.5..10-1-gcc-2.9.6'
        ]

    configure.petsc_configure(configure_options)
