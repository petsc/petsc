#include <../src/ksp/ksp/impls/gmres/gmresimpl.h> /*I  "petscksp.h"  I*/

/*@
  KSPGMRESSetPreAllocateVectors - Causes `KSPGMRES` and `KSPFGMRES` to preallocate all its
  needed work vectors at initial setup rather than the default, which
  is to allocate several at a time when needed.

  Logically Collective

  Input Parameter:
. ksp - iterative context obtained from `KSPCreate()`

  Options Database Key:
. -ksp_gmres_preallocate - Activates `KSPGmresSetPreAllocateVectors()`

  Level: intermediate

  Notes:
  If one knows the number of iterations will be greater than or equal to the `KSPGMRESSetRestart()` size then calling
  this routine can result in faster performance since it minimizes the number of separate memory allocations used
  and can improve the performance of `VecMDot()` and `VecMAXPY()` which may utilize BLAS 2 operations that benefit from
  the larger allocations.

  Using this function with vectors in GPU memory may waste GPU memory if not all the restart directions are used in solving the system,
  that is the solver converges before the number of iterations reaches the restart value.

.seealso: [](ch_ksp), `KSPGMRESSetRestart()`, `KSPOrthogonalizationSet()`, `KSPOrthogonalizationGet()`,
          `VecMDot()`, `VecMAXPY()`
@*/
PetscErrorCode KSPGMRESSetPreAllocateVectors(KSP ksp)
{
  PetscFunctionBegin;
  PetscTryMethod(ksp, "KSPGMRESSetPreAllocateVectors_C", (KSP), (ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  KSPGMRESSetVectorAllocationSize - Set the number of Krylov search vectors allocated at a time by `KSPGMRES`, `KSPFGMRES`, etc.

  Logically Collective

  Input Parameters:
+ ksp - iterative context obtained from `KSPCreate()`
- n   - number of vectors, default 10

  Options Database Key:
. -ksp_gmres_vector_allocation_size n - number of vectors to allocate each time

  Level: intermediate

  Notes:
  Use `PETSC_CURRENT` to retain the current value and `PETSC_DETERMINE` to use the default value.

  It is usually much more efficient to allocate GPU vectors in large batches.

.seealso: [](ch_ksp), `KSPGMRES`, `KSPGMRESSetPreAllocateVectors()`
@*/
PetscErrorCode KSPGMRESSetVectorAllocationSize(KSP ksp, PetscInt n)
{
  PetscFunctionBegin;
  PetscValidLogicalCollectiveInt(ksp, n, 2);
  PetscTryMethod(ksp, "KSPGMRESSetVectorAllocationSize_C", (KSP, PetscInt), (ksp, n));
  PetscFunctionReturn(PETSC_SUCCESS);
}
