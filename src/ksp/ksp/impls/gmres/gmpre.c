#define PETSCKSP_DLL

#include "../src/ksp/ksp/impls/gmres/gmresimpl.h"       /*I  "petscksp.h"  I*/

#undef __FUNCT__  
#define __FUNCT__ "KSPGMRESSetPreAllocateVectors" 
/*@
    KSPGMRESSetPreAllocateVectors - Causes GMRES and FGMRES to preallocate all its
    needed work vectors at initial setup rather than the default, which 
    is to allocate them in chunks when needed.

    Collective on KSP

    Input Parameter:
.   ksp   - iterative context obtained from KSPCreate

    Options Database Key:
.   -ksp_gmres_preallocate - Activates KSPGmresSetPreAllocateVectors()

    Level: intermediate

.keywords: GMRES, preallocate, vectors

.seealso: KSPGMRESSetRestart(), KSPGMRESSetOrthogonalization()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT KSPGMRESSetPreAllocateVectors(KSP ksp)
{
  PetscErrorCode ierr,(*f)(KSP);

  PetscFunctionBegin;
  ierr = PetscObjectQueryFunction((PetscObject)ksp,"KSPGMRESSetPreAllocateVectors_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(ksp);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

