#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    configure_options = [
        '--can-execute=0',
        '--sizeof_void_p=4',
        '--sizeof_short=2',
        '--sizeof_int=4',
        '--sizeof_long=4',
        '--sizeof_long_long=8',
        '--sizeof_float=4',
        '--sizeof_double=8',
        '--bits_per_byte=8',
        '--sizeof_MPI_Comm=4',
        '--sizeof_MPI_Fint=4'
    ]

    configure.petsc_configure(configure_options)
