#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

#ifdef PETSC_HAVE_LIBCEED
  #include <petscfeceed.h>

/*@C
  PetscFESetCeed - Set the `Ceed` object to a `PetscFE`

  Not Collective

  Input Parameters:
+ fe   - The `PetscFE`
- ceed - The `Ceed` object

  Level: intermediate

.seealso: `PetscFE`, `PetscFEGetCeedBasis()`, `DMGetCeed()`
@*/
PetscErrorCode PetscFESetCeed(PetscFE fe, Ceed ceed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe, PETSCFE_CLASSID, 1);
  if (fe->ceed == ceed) PetscFunctionReturn(0);
  PetscCallCEED(CeedReferenceCopy(ceed, &fe->ceed));
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGetCeedBasis - Get the `Ceed` object mirroring this `PetscFE`

  Not Collective

  Input Parameter:
. fe - The `PetscFE`

  Output Parameter:
. basis - The `CeedBasis`

  Level: intermediate

  Note:
  This is a borrowed reference, so it is not freed.

.seealso: `PetscFE`, `PetscFESetCeed()`, `DMGetCeed()`
@*/
PetscErrorCode PetscFEGetCeedBasis(PetscFE fe, CeedBasis *basis)
{
  PetscSpace      sp;
  PetscQuadrature q;
  PetscInt        dim, Nc, deg, ord;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe, PETSCFE_CLASSID, 1);
  PetscValidBoolPointer(basis, 2);
  if (!fe->ceedBasis && fe->ceed) {
    PetscCall(PetscFEGetSpatialDimension(fe, &dim));
    PetscCall(PetscFEGetNumComponents(fe, &Nc));
    PetscCall(PetscFEGetBasisSpace(fe, &sp));
    PetscCall(PetscSpaceGetDegree(sp, &deg, NULL));
    PetscCall(PetscFEGetQuadrature(fe, &q));
    PetscCall(PetscQuadratureGetOrder(q, &ord));
    PetscCallCEED(CeedBasisCreateTensorH1Lagrange(fe->ceed, dim, Nc, deg + 1, (ord + 1) / 2, CEED_GAUSS, &fe->ceedBasis));
  }
  *basis = fe->ceedBasis;
  PetscFunctionReturn(0);
}

#endif
