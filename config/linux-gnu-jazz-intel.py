#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    configure_options = [
        '--with-mpi-dir=/soft/apps/packages/mpich-gm-1.2.5..10-1-intel-7.1',
        '--with-blas-lapack-dir=/soft/com/packages/intel-7.1/mkl60'
        ]

    configure.petsc_configure(configure_options)
