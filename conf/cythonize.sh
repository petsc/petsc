#!/bin/sh
cython --cleanup 3 -w src -Iinclude $@ petsc4py.PETSc.pyx && \
cython --cleanup 3 -w src -Iinclude $@ libpetsc4py/libpetsc4py.pyx && \
mv src/petsc4py.PETSc*.h src/include/petsc4py
