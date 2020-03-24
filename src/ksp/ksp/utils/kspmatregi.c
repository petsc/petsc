#include <petsc/private/kspimpl.h> /*I "petscsnes.h" I*/
#include <../src/ksp/ksp/utils/schurm/schurm.h>
#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

static PetscBool KSPMatRegisterAllCalled = PETSC_FALSE;

/*@C
  KSPMatRegisterAll - Registers all matrix implementations in the KSP package.

  Not Collective

  Level: advanced

.seealso: MatRegisterAll(),  KSPInitializePackage()
@*/
PetscErrorCode KSPMatRegisterAll(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (KSPMatRegisterAllCalled) PetscFunctionReturn(0);
  KSPMatRegisterAllCalled = PETSC_TRUE;
  ierr = MatRegister(MATSCHURCOMPLEMENT,       MatCreate_SchurComplement);CHKERRQ(ierr);
  ierr = MatRegister(MATLMVMDFP,               MatCreate_LMVMDFP);CHKERRQ(ierr);
  ierr = MatRegister(MATLMVMBFGS,              MatCreate_LMVMBFGS);CHKERRQ(ierr);
  ierr = MatRegister(MATLMVMSR1,               MatCreate_LMVMSR1);CHKERRQ(ierr);
  ierr = MatRegister(MATLMVMBROYDEN,           MatCreate_LMVMBrdn);CHKERRQ(ierr);
  ierr = MatRegister(MATLMVMBADBROYDEN,        MatCreate_LMVMBadBrdn);CHKERRQ(ierr);
  ierr = MatRegister(MATLMVMSYMBROYDEN,        MatCreate_LMVMSymBrdn);CHKERRQ(ierr);
  ierr = MatRegister(MATLMVMSYMBADBROYDEN,     MatCreate_LMVMSymBadBrdn);CHKERRQ(ierr);
  ierr = MatRegister(MATLMVMDIAGBROYDEN,       MatCreate_LMVMDiagBrdn);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
