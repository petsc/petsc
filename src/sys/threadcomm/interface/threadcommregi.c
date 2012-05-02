 
#include <petsc-private/threadcommimpl.h>     /*I    "petscthreadcomm.h"  I*/

EXTERN_C_BEGIN
extern PetscErrorCode PetscThreadCommCreate_NoThread(PetscThreadComm);
#if defined(PETSC_HAVE_PTHREADCLASSES)
extern PetscErrorCode PetscThreadCommCreate_PThread(PetscThreadComm);
#endif
#if defined(PETSC_HAVE_OPENMP)
extern PetscErrorCode PetscThreadCommCreate_OpenMP(PetscThreadComm);
#endif
EXTERN_C_END

extern PetscBool PetscThreadCommRegisterAllCalled;

#undef __FUNCT__
#define __FUNCT__ "PetscThreadCommRegisterAll"
/*@C
   PetscThreadCommRegisterAll - Registers of all the thread communicator models

   Not Collective

   Level: advanced

.keywords: PetscThreadComm, register, all

.seealso: PetscThreadCommRegisterDestroy()
@*/
PetscErrorCode PetscThreadCommRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscThreadCommRegisterAllCalled = PETSC_TRUE;
  ierr = PetscThreadCommRegisterDynamic(NOTHREAD,         path,"PetscThreadCommCreate_NoThread",         PetscThreadCommCreate_NoThread);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PTHREADCLASSES)
  ierr = PetscThreadCommRegisterDynamic(PTHREAD,          path,"PetscThreadCommCreate_PThread",          PetscThreadCommCreate_PThread);CHKERRQ(ierr);
#endif
#if defined(PETSC_HAVE_OPENMP)
  ierr = PetscThreadCommRegisterDynamic(OPENMP,         path,"PetscThreadCommCreate_OpenMP",         PetscThreadCommCreate_OpenMP);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}
