#!/bin/sh
dirn=$(dirname "$0")
topdir=$(cd "$dirn"/.. && pwd)
python"${py:=}" "$topdir/conf/cythonize.py" \
    --working "$topdir/src" "$@" \
    "petsc4py/PETSc.pyx"
