#include <petsc.h>
/*
#include "libpetsc4py/libpetsc4py.h"
*/
PETSC_EXTERN int import_libpetsc4py(void);
PETSC_EXTERN PetscErrorCode MatPythonGetContext(Mat,void**);
PETSC_EXTERN PetscErrorCode MatPythonSetContext(Mat,void*);
PETSC_EXTERN PetscErrorCode PCPythonGetContext(PC,void**);
PETSC_EXTERN PetscErrorCode PCPythonSetContext(PC,void*);
PETSC_EXTERN PetscErrorCode KSPPythonGetContext(KSP,void**);
PETSC_EXTERN PetscErrorCode KSPPythonSetContext(KSP,void*);
PETSC_EXTERN PetscErrorCode SNESPythonGetContext(SNES,void**);
PETSC_EXTERN PetscErrorCode SNESPythonSetContext(SNES,void*);
PETSC_EXTERN PetscErrorCode TSPythonGetContext(TS,void**);
PETSC_EXTERN PetscErrorCode TSPythonSetContext(TS,void*);
PETSC_EXTERN PetscErrorCode TaoPythonGetContext(Tao,void**);
PETSC_EXTERN PetscErrorCode TaoPythonSetContext(Tao,void*);
PETSC_EXTERN PetscErrorCode PetscPythonRegisterAll(void);
