#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <petsc.h>
#include "libpetsc4py/libpetsc4py.c"
