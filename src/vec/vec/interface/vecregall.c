
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
#if defined(PETSC_HAVE_VIENNACL)
PETSC_EXTERN PetscErrorCode VecCreate_SeqViennaCL(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_MPIViennaCL(Vec);
PETSC_EXTERN PetscErrorCode VecCreate_ViennaCL(Vec);
#endif
#if 0
#if defined(PETSC_HAVE_SIEVE)
PETSC_EXTERN PetscErrorCode VecCreate_Sieve(Vec);
#endif
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
#if defined PETSC_HAVE_VIENNACL
  ierr = VecRegisterDynamic(VECSEQVIENNACL,  path, "VecCreate_SeqViennaCL",  VecCreate_SeqViennaCL);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECMPIVIENNACL,  path, "VecCreate_MPIViennaCL",  VecCreate_MPIViennaCL);CHKERRQ(ierr);
  ierr = VecRegisterDynamic(VECVIENNACL,     path, "VecCreate_ViennaCL",     VecCreate_ViennaCL);CHKERRQ(ierr);
#endif
#if 0
#if defined(PETSC_HAVE_SIEVE)
  ierr = VecRegister(VECSIEVE,      VecCreate_Sieve);CHKERRQ(ierr);
#endif
#endif
  PetscFunctionReturn(0);
}

