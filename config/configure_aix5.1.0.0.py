#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    configure_options = [
        '-PETSC_ARCH='+configure.getarch()
        ]

    configure.petsc_configure(configure_options)
