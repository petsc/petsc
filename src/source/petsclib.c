#include <petsc.h>

#if (PETSC_VERSION_MAJOR    == 2 && \
     PETSC_VERSION_MINOR    == 3 && \
     PETSC_VERSION_SUBMINOR == 2 && \
     PETSC_VERSION_RELEASE  == 1)
#define PETSCMAT_DLL
#include "src/mat/impls/is/matis.c"
#undef  PETSCMAT_DLL
#endif

#define PETSCMAT_DLL
#include "src/mat/impls/python/python.c"
#undef  PETSCMAT_DLL

#define PETSCKSP_DLL
#include "src/ksp/pc/impls/schur/schur.c"
#include "src/ksp/pc/impls/python/python.c"
#undef PETSCKSP_DLL

#define PETSCKSP_DLL
#include "src/ksp/ksp/impls/python/python.c"
#undef PETSCKSP_DLL

#define PETSCSNES_DLL
#include "src/snes/impls/python/python.c"
#undef PETSCSNES_DLL

#define PETSCTS_DLL
#include "src/ts/impls/implicit/user/user.c"
#include "src/ts/impls/python/python.c"
#undef PETSCTS_DLL
