
#include <../src/ksp/ksp/impls/gmres/gmresimpl.h>       /*I  "petscksp.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetPreAllocateVectors" 
/*@
    KSPGMRESSetPreAllocateVectors - Causes GMRES and FGMRES to preallocate all its
    needed work vectors at initial setup rather than the default, which 
    is to allocate them in chunks when needed.

    Logically Collective on KSP

    Input Parameter:
.   ksp   - iterative context obtained from KSPCreate

    Options Database Key:
.   -ksp_gmres_preallocate - Activates KSPGmresSetPreAllocateVectors()

    Level: intermediate

.keywords: GMRES, preallocate, vectors

.seealso: KSPGMRESSetRestart(), KSPGMRESSetOrthogonalization(), KSPGMRESGetOrthogonalization()
@*/
PetscErrorCode  KSPGMRESSetPreAllocateVectors(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscTryMethod(ksp,"KSPGMRESSetPreAllocateVectors_C",(KSP),(ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

