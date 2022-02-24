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
  PetscFunctionBegin;
  if (KSPMatRegisterAllCalled) PetscFunctionReturn(0);
  KSPMatRegisterAllCalled = PETSC_TRUE;
  CHKERRQ(MatRegister(MATSCHURCOMPLEMENT,       MatCreate_SchurComplement));
  CHKERRQ(MatRegister(MATLMVMDFP,               MatCreate_LMVMDFP));
  CHKERRQ(MatRegister(MATLMVMBFGS,              MatCreate_LMVMBFGS));
  CHKERRQ(MatRegister(MATLMVMSR1,               MatCreate_LMVMSR1));
  CHKERRQ(MatRegister(MATLMVMBROYDEN,           MatCreate_LMVMBrdn));
  CHKERRQ(MatRegister(MATLMVMBADBROYDEN,        MatCreate_LMVMBadBrdn));
  CHKERRQ(MatRegister(MATLMVMSYMBROYDEN,        MatCreate_LMVMSymBrdn));
  CHKERRQ(MatRegister(MATLMVMSYMBADBROYDEN,     MatCreate_LMVMSymBadBrdn));
  CHKERRQ(MatRegister(MATLMVMDIAGBROYDEN,       MatCreate_LMVMDiagBrdn));
  PetscFunctionReturn(0);
}
