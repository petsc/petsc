#include <petsc/private/kspimpl.h> /*I "petscsnes.h" I*/
#include <../src/ksp/ksp/utils/schurm/schurm.h>
#include <../src/ksp/ksp/utils/lmvm/lmvm.h>

static PetscBool KSPMatRegisterAllCalled = PETSC_FALSE;

/*@C
  KSPMatRegisterAll - Registers all matrix implementations in the KSP package.

  Not Collective

  Level: advanced

.seealso: `MatRegisterAll()`, `KSPInitializePackage()`
@*/
PetscErrorCode KSPMatRegisterAll(void)
{
  PetscFunctionBegin;
  if (KSPMatRegisterAllCalled) PetscFunctionReturn(0);
  KSPMatRegisterAllCalled = PETSC_TRUE;
  PetscCall(MatRegister(MATSCHURCOMPLEMENT,       MatCreate_SchurComplement));
  PetscCall(MatRegister(MATLMVMDFP,               MatCreate_LMVMDFP));
  PetscCall(MatRegister(MATLMVMBFGS,              MatCreate_LMVMBFGS));
  PetscCall(MatRegister(MATLMVMSR1,               MatCreate_LMVMSR1));
  PetscCall(MatRegister(MATLMVMBROYDEN,           MatCreate_LMVMBrdn));
  PetscCall(MatRegister(MATLMVMBADBROYDEN,        MatCreate_LMVMBadBrdn));
  PetscCall(MatRegister(MATLMVMSYMBROYDEN,        MatCreate_LMVMSymBrdn));
  PetscCall(MatRegister(MATLMVMSYMBADBROYDEN,     MatCreate_LMVMSymBadBrdn));
  PetscCall(MatRegister(MATLMVMDIAGBROYDEN,       MatCreate_LMVMDiagBrdn));
  PetscFunctionReturn(0);
}
