
#include <../src/mat/impls/aij/seq/aij.h>

#if defined(PETSC_HAVE_HDF5)
PetscErrorCode MatLoad_AIJ_HDF5(Mat mat, PetscViewer viewer)
{
  /*PetscErrorCode ierr;*/

  PetscFunctionBegin;
  SETERRQ(PetscObjectComm((PetscObject)mat),PETSC_ERR_SUP,"MatLoad_MPIAIJ_HDF5 is not implemented yet!");
  PetscFunctionReturn(0);
}
#endif

