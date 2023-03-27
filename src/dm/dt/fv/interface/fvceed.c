#include <petsc/private/petscfvimpl.h> /*I "petscfv.h" I*/

#ifdef PETSC_HAVE_LIBCEED
  #include <petscfvceed.h>

/*@C
  PetscFVSetCeed - Set the `Ceed` object to a `PetscFV`

  Not Collective

  Input Parameters:
+ fv   - The `PetscFV`
- ceed - The `Ceed` object

  Level: intermediate

.seealso: `PetscFV`, `PetscFVGetCeedBasis()`, `DMGetCeed()`
@*/
PetscErrorCode PetscFVSetCeed(PetscFV fv, Ceed ceed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fv, PETSCFV_CLASSID, 1);
  if (fv->ceed == ceed) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCallCEED(CeedReferenceCopy(ceed, &fv->ceed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  PetscFVGetCeedBasis - Get the `Ceed` object mirroring this `PetscFV`

  Not Collective

  Input Parameter:
. fv - The `PetscFV`

  Output Parameter:
. basis - The `CeedBasis`

  Level: intermediate

  Note:
  This is a borrowed reference, so it is not freed.

.seealso: `PetscFV`, `PetscFVSetCeed()`, `DMGetCeed()`
@*/
PetscErrorCode PetscFVGetCeedBasis(PetscFV fv, CeedBasis *basis)
{
  PetscQuadrature q;
  PetscInt        dim, Nc, ord;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fv, PETSCFV_CLASSID, 1);
  PetscValidBoolPointer(basis, 2);
  if (!fv->ceedBasis && fv->ceed) {
    PetscCall(PetscFVGetSpatialDimension(fv, &dim));
    PetscCall(PetscFVGetNumComponents(fv, &Nc));
    PetscCall(PetscFVGetQuadrature(fv, &q));
    PetscCall(PetscQuadratureGetOrder(q, &ord));
    PetscCallCEED(CeedBasisCreateTensorH1Lagrange(fv->ceed, dim, Nc, 1, (ord + 1) / 2, CEED_GAUSS, &fv->ceedBasis));
  }
  *basis = fv->ceedBasis;
  PetscFunctionReturn(PETSC_SUCCESS);
}

#endif
