
#include <petscvec.h>
PETSC_EXTERN_C PetscErrorCode  VecCreate_Seq(Vec);
PETSC_EXTERN_C PetscErrorCode  VecCreate_MPI(Vec);
PETSC_EXTERN_C PetscErrorCode  VecCreate_Standard(Vec);
PETSC_EXTERN_C PetscErrorCode  VecCreate_Shared(Vec);
#if defined(PETSC_HAVE_CUSP)
PETSC_EXTERN_C PetscErrorCode  VecCreate_SeqCUSP(Vec);
PETSC_EXTERN_C PetscErrorCode  VecCreate_MPICUSP(Vec);
PETSC_EXTERN_C PetscErrorCode  VecCreate_CUSP(Vec);
#endif
#if 0
#if defined(PETSC_HAVE_SIEVE)
PETSC_EXTERN_C PetscErrorCode  VecCreate_Sieve(Vec);
#endif
#endif

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

