#!/usr/bin/env python

if __name__ == '__main__':
    import configure

    configure_options = [
         '--enable-dynamic=0'
        ]

    configure.petsc_configure(configure_options)
