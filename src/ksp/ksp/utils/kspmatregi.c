#include <petsc/private/kspimpl.h> /*I "petscsnes.h" I*/
#include <../src/ksp/ksp/utils/schurm/schurm.h>
#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

static PetscBool KSPMatRegisterAllCalled = PETSC_FALSE;

/*@C
  KSPMatRegisterAll - Registers all matrix implementations in the KSP package.

  Not Collective

  Level: advanced

.keywords: Mat, KSP, register, all

.seealso: MatRegisterAll(),  KSPInitializePackage()
@*/
PetscErrorCode KSPMatRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (KSPMatRegisterAllCalled) PetscFunctionReturn(0);
  KSPMatRegisterAllCalled = PETSC_TRUE;
  ierr = MatRegister(MATSCHURCOMPLEMENT, MatCreate_SchurComplement);CHKERRQ(ierr);
  ierr = MatRegister(MATLDFP,            MatCreate_LDFP);CHKERRQ(ierr);
  ierr = MatRegister(MATLBFGS,           MatCreate_LBFGS);CHKERRQ(ierr);
  ierr = MatRegister(MATLSR1,            MatCreate_LSR1);CHKERRQ(ierr);
  ierr = MatRegister(MATLBRDN,           MatCreate_LBRDN);CHKERRQ(ierr);
  ierr = MatRegister(MATLMBRDN,          MatCreate_LMBRDN);CHKERRQ(ierr);
  ierr = MatRegister(MATLSBRDN,          MatCreate_LSBRDN);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}