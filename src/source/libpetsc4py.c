/* ---------------------------------------------------------------- */

#include <Python.h>

/* ---------------------------------------------------------------- */

#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1

#include <petsc.h>

/* ---------------------------------------------------------------- */

#if !defined(PETSC_VERSION_)
#define PETSC_VERSION_(MAJOR,MINOR,SUBMINOR)   \
       (PETSC_VERSION_MAJOR    == MAJOR    && \
	PETSC_VERSION_MINOR    == MINOR    && \
	PETSC_VERSION_SUBMINOR == SUBMINOR && \
	PETSC_VERSION_RELEASE  == 1)
#endif

/* ---------------------------------------------------------------- */

#define PETSCMAT_DLL
#include "src/mat/impls/python/python.c"
#undef  PETSCMAT_DLL

#define PETSCKSP_DLL
#include "src/ksp/pc/impls/python/python.c"
#undef PETSCKSP_DLL

#define PETSCKSP_DLL
#include "src/ksp/ksp/impls/python/python.c"
#undef PETSCKSP_DLL

#define PETSCSNES_DLL
#include "src/snes/impls/python/python.c"
#undef PETSCSNES_DLL

#define PETSCTS_DLL
#include "src/ts/impls/python/python.c"
#undef PETSCTS_DLL

/* ---------------------------------------------------------------- */
