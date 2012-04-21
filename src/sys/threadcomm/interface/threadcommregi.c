
#include <petsc-private/threadcommimpl.h>     /*I    "petscthreadcomm.h"  I*/

#if defined(PETSC_HAVE_PTHREADCLASSES)
EXTERN_C_BEGIN
extern PetscErrorCode PetscThreadCommCreate_PThread(PetscThreadComm);
EXTERN_C_END
#endif

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
#if defined(PETSC_HAVE_PTHREADCLASSES)
  ierr = PetscThreadCommRegisterDynamic(PTHREAD,          path,"PetscThreadCommCreate_PThread",          PetscThreadCommCreate_PThread);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}
