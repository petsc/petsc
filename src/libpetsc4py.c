#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1
#include <Python.h>
#include <petsc.h>
#if PETSC_VERSION_(3,1,0) || PETSC_VERSION_(3,0,0)
  #include "source/libpetsc4py.c"
#else
  #include "libpetsc4py/libpetsc4py.c"
#endif
