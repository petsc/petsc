#define PETSCVEC_DLL

#include "private/vecimpl.h"     /*I  "vec.h"  I*/
EXTERN_C_BEGIN
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Seq(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_MPI(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Shared(Vec);
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_FETI(Vec);
#if 0
#if defined(PETSC_HAVE_SIEVE)
EXTERN PetscErrorCode PETSCVEC_DLLEXPORT VecCreate_Sieve(Vec);
#endif
#endif
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "VecRegisterAll"
/*@C
  VecRegisterAll - Registers all of the vector components in the Vec package.

  Not Collective

  Input parameter:
. path - The dynamic library path

  Level: advanced

.keywords: Vec, register, all
.seealso:  VecRegister(), VecRegisterDestroy(), VecRegisterDynamic()
@*/
PetscErrorCode PETSCVEC_DLLEXPORT VecRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  VecRegisterAllCalled = PETSC_TRUE;

  ierr = VecRegisterDynamic(VECSEQ,      path, "VecCreate_Seq",      VecCreate_Seq);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECMPI,      path, "VecCreate_MPI",      VecCreate_MPI);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECSHARED,   path, "VecCreate_Shared",   VecCreate_Shared);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECFETI,     path, "VecCreate_FETI",     VecCreate_FETI);CHKERRQ(ierr);
#if 0
#if defined(PETSC_HAVE_SIEVE)
  ierr = VecRegisterDynamic(VECSIEVE,    path, "VecCreate_Sieve",    VecCreate_Sieve);CHKERRQ(ierr);
#endif
#endif
  PetscFunctionReturn(0);
}

