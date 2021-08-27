#include "../cupmcontext.hpp" /*I "petscdevice.h" I*/

PetscErrorCode PetscDeviceContextCreate_CUDA(PetscDeviceContext dctx)
{
  static const Petsc::CUPMContextCuda  contextCuda;
  PetscDeviceContext_(CUDA)           *dci;
  PetscErrorCode                       ierr;

  PetscFunctionBegin;
  ierr = PetscNew(&dci);CHKERRQ(ierr);
  dctx->data = static_cast<void*>(dci);
  ierr = PetscMemcpy(dctx->ops,&contextCuda.ops,sizeof(contextCuda.ops));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
