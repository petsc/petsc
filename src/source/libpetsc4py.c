/* ---------------------------------------------------------------- */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#if PY_VERSION_HEX < 0x02050000
typedef int Py_ssize_t;
#endif

/* ---------------------------------------------------------------- */

#define MPICH_SKIP_MPICXX 1
#define OMPI_SKIP_MPICXX 1

#include <petsc.h>

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
