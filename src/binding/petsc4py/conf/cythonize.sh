#!/bin/sh
python -m cython --3str --cleanup 3 -w src -Iinclude $@ petsc4py.PETSc.pyx -o petsc4py.PETSc.c && \
mv src/petsc4py.PETSc*.h src/include/petsc4py
