
#ifndef __NOTHREADIMPLH
#define __NOTHREADIMPLH

#include <petsc-private/threadcommimpl.h>

EXTERN_C_BEGIN
extern PetscErrorCode PetscThreadCommCreate_NoThread(PetscThreadComm);
EXTERN_C_END

/* extern PetscErrorCode PetscThreadCommRunKernel_NoThread(MPI_Comm,PetscThreadCommJobCtx); */

#endif
