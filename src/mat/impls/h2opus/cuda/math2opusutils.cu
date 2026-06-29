#include <petsc/private/matimpl.h>
#include <petsc/private/vecimpl.h>
#include <petscsf.h>
#if defined(PETSC_HAVE_CUDA)
  #include <thrust/for_each.h>
  #include <thrust/device_vector.h>
  #include <thrust/execution_policy.h>
#endif

PETSC_INTERN PetscErrorCode MatDenseGetH2OpusStridedSF(Mat A, PetscSF h2sf, PetscSF *osf)
{
  PetscSF asf;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(A, MAT_CLASSID, 1);
  PetscValidHeaderSpecific(h2sf, PETSCSF_CLASSID, 2);
  PetscAssertPointer(osf, 3);
  PetscCall(PetscObjectQuery((PetscObject)A, "_math2opus_stridedsf", (PetscObject *)&asf));
  if (!asf) {
    PetscInt lda;

    PetscCall(MatDenseGetLDA(A, &lda));
    PetscCall(PetscSFCreateStridedSF(h2sf, A->cmap->N, lda, PETSC_DECIDE, &asf));
    PetscCall(PetscObjectCompose((PetscObject)A, "_math2opus_stridedsf", (PetscObject)asf));
    PetscCall(PetscObjectDereference((PetscObject)asf));
  }
  *osf = asf;
  PetscFunctionReturn(PETSC_SUCCESS);
}
