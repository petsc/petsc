#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    configure_options = [
        # cannot build shared libraries are our particular test machine
         '--with-shared=0'
        ]

    configure.petsc_configure(configure_options)
