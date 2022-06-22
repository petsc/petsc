
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>       /*I  "petscksp.h"  I*/

/*@
    KSPGMRESSetPreAllocateVectors - Causes GMRES and FGMRES to preallocate all its
    needed work vectors at initial setup rather than the default, which
    is to allocate them in chunks when needed.

    Logically Collective on ksp

    Input Parameter:
.   ksp   - iterative context obtained from KSPCreate

    Options Database Key:
.   -ksp_gmres_preallocate - Activates KSPGmresSetPreAllocateVectors()

    Level: intermediate

.seealso: `KSPGMRESSetRestart()`, `KSPGMRESSetOrthogonalization()`, `KSPGMRESGetOrthogonalization()`
@*/
PetscErrorCode  KSPGMRESSetPreAllocateVectors(KSP ksp)
{
  PetscFunctionBegin;
  PetscTryMethod(ksp,"KSPGMRESSetPreAllocateVectors_C",(KSP),(ksp));
  PetscFunctionReturn(0);
}
