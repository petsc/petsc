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

#if (PETSC_VERSION_(3,1,0) || \
     PETSC_VERSION_(3,0,0) || \
     PETSC_VERSION_(2,3,3) || \
     PETSC_VERSION_(2,3,2))
#define IS_CLASSID    IS_COOKIE
#define VEC_CLASSID   VEC_COOKIE
#define MAT_CLASSID   MAT_COOKIE
#define KSP_CLASSID   KSP_COOKIE
#define PC_CLASSID    PC_COOKIE
#define SNES_CLASSID  SNES_COOKIE
#define TS_CLASSID    TS_COOKIE
#endif

/* ---------------------------------------------------------------- */

#include "python_core.h"

#define PETSCMAT_DLL
#include "python_mat.c"
#undef  PETSCMAT_DLL

#define PETSCKSP_DLL
#include "python_ksp.c"
#undef PETSCKSP_DLL

#define PETSCKSP_DLL
#include "python_pc.c"
#undef PETSCKSP_DLL

#define PETSCSNES_DLL
#include "python_snes.c"
#undef PETSCSNES_DLL

#define PETSCTS_DLL
#include "python_ts.c"
#undef PETSCTS_DLL

/* ---------------------------------------------------------------- */
