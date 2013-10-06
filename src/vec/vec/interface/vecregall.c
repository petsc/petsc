
#include <petscvec.h>
PETSC_EXTERN PetscErrorCode VecCreate_Seq(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_MPI(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_Standard(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_Shared(Vec);
#if defined(PETSC_HAVE_CUSP)
PETSC_EXTERN PetscErrorCode VecCreate_SeqCUSP(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_MPICUSP(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_CUSP(Vec);
#endif

#undef __FUNCT__
#define __FUNCT__ "VecRegisterAll"
/*@C
  VecRegisterAll - Registers all of the vector components in the Vec package.

  Not Collective

  Level: advanced

.keywords: Vec, register, all
.seealso:  VecRegister(), VecRegisterDestroy(), VecRegister()
@*/
PetscErrorCode  VecRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  VecRegisterAllCalled = PETSC_TRUE;

  ierr = VecRegister(VECSEQ,        VecCreate_Seq);CHKERRQ(ierr);
  ierr = VecRegister(VECMPI,        VecCreate_MPI);CHKERRQ(ierr);
  ierr = VecRegister(VECSTANDARD,   VecCreate_Standard);CHKERRQ(ierr);
  ierr = VecRegister(VECSHARED,     VecCreate_Shared);CHKERRQ(ierr);
#if defined PETSC_HAVE_CUSP
  ierr = VecRegister(VECSEQCUSP,    VecCreate_SeqCUSP);CHKERRQ(ierr);
  ierr = VecRegister(VECMPICUSP,    VecCreate_MPICUSP);CHKERRQ(ierr);
  ierr = VecRegister(VECCUSP,       VecCreate_CUSP);CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

