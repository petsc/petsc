#!/bin/sh
cython --cleanup 3 -w src -Iinclude $@ petsc4py.PETSc.pyx && \
mv src/petsc4py.PETSc*.h src/include/petsc4py
