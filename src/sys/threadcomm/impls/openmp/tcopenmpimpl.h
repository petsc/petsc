
#if !defined(__TCOPENMPIMPLH)
#define __TCOPENMPIMPLH

#include <petsc/private/threadcommimpl.h>
PETSC_EXTERN PetscErrorCode PetscThreadCommCreate_OpenMP(PetscThreadComm);
extern PetscErrorCode PetscThreadCommRunKernel_OpenMP(PetscThreadComm,PetscThreadCommJobCtx);

#endif
