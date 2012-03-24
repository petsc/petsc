
#include <petsc-private/vecimpl.h>     /*I  "vec.h"  I*/
EXTERN_C_BEGIN
extern PetscErrorCode  VecCreate_Seq(Vec);
extern PetscErrorCode  VecCreate_MPI(Vec);
extern PetscErrorCode  VecCreate_Standard(Vec);
extern PetscErrorCode  VecCreate_Shared(Vec);
#if defined(PETSC_HAVE_PTHREADCLASSES)
extern PetscErrorCode  VecCreate_SeqPThread(Vec);
extern PetscErrorCode  VecCreate_MPIPThread(Vec);
extern PetscErrorCode  VecCreate_PThread(Vec);
#endif
#if defined(PETSC_HAVE_CUSP)
extern PetscErrorCode  VecCreate_SeqCUSP(Vec);
extern PetscErrorCode  VecCreate_MPICUSP(Vec);
extern PetscErrorCode  VecCreate_CUSP(Vec);
#endif
#if 0
#if defined(PETSC_HAVE_SIEVE)
extern PetscErrorCode  VecCreate_Sieve(Vec);
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
PetscErrorCode  VecRegisterAll(const char path[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  VecRegisterAllCalled = PETSC_TRUE;

  ierr = VecRegisterDynamic(VECSEQ,       path, "VecCreate_Seq",       VecCreate_Seq);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECMPI,       path, "VecCreate_MPI",       VecCreate_MPI);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECSTANDARD,  path, "VecCreate_Standard",  VecCreate_Standard);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECSHARED,    path, "VecCreate_Shared",    VecCreate_Shared);CHKERRQ(ierr);
#if defined(PETSC_HAVE_PTHREADCLASSES)
  ierr = VecRegisterDynamic(VECSEQPTHREAD,path, "VecCreate_SeqPThread",   VecCreate_SeqPThread);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECMPIPTHREAD,path, "VecCreate_MPIPThread",  VecCreate_MPIPThread);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECPTHREAD,   path, "VecCreate_PThread",      VecCreate_PThread);CHKERRQ(ierr);
#endif
#if defined PETSC_HAVE_CUSP
  ierr = VecRegisterDynamic(VECSEQCUSP,  path, "VecCreate_SeqCUSP",  VecCreate_SeqCUSP);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECMPICUSP,  path, "VecCreate_MPICUSP",  VecCreate_MPICUSP);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECCUSP,     path, "VecCreate_CUSP",     VecCreate_CUSP);CHKERRQ(ierr);
#endif
#if 0
#if defined(PETSC_HAVE_SIEVE)
  ierr = VecRegisterDynamic(VECSIEVE,    path, "VecCreate_Sieve",    VecCreate_Sieve);CHKERRQ(ierr);
#endif
#endif
  PetscFunctionReturn(0);
}

