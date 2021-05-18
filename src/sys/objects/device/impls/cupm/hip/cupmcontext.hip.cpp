#include "../cupmcontext.hpp" /*I "petscdevice.h" I*/

PetscErrorCode PetscDeviceContextCreate_HIP(PetscDeviceContext dctx)
{
  static const Petsc::CUPMContextHip  contextHip;
  PetscDeviceContext_(HIP)           *dci;
  PetscErrorCode                      ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = static_cast<void*>(dci);
  ierr = PetscMemcpy(dctx->ops,&contextHip.ops,sizeof(contextHip.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
