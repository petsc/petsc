#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    # build on harley
    configure_options = [
        '-PETSC_ARCH=aix5.1.0.0',
        '-PETSC_DIR=/usr/common/homes/k/kaushik/tmp/petsc-test'
        ]

    configure.petsc_configure(configure_options)
