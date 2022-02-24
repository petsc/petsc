#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/

#ifdef PETSC_HAVE_LIBCEED
#include <petscfeceed.h>

/*@C
  PetscFESetCeed - Set the Ceed object

  Not Collective

  Input Parameters:
+ fe   - The PetscFE
- ceed - The Ceed object

  Level: intermediate

.seealso: PetscFEGetCeedBasis(), DMGetCeed()
@*/
PetscErrorCode PetscFESetCeed(PetscFE fe, Ceed ceed)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe, PETSCFE_CLASSID, 1);
  if (fe->ceed == ceed) PetscFunctionReturn(0);
  CHKERRQ_CEED(CeedReferenceCopy(ceed, &fe->ceed));
  PetscFunctionReturn(0);
}

/*@C
  PetscFEGetCeedBasis - Get the Ceed object mirroring this FE

  Not Collective

  Input Parameter:
. fe - The PetscFE

  Output Parameter:
. basis - The CeedBasis

  Note: This is a borrowed reference, so it is not freed.

  Level: intermediate

.seealso: PetscFESetCeed(), DMGetCeed()
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
    CHKERRQ(PetscFEGetSpatialDimension(fe, &dim));
    CHKERRQ(PetscFEGetNumComponents(fe, &Nc));
    CHKERRQ(PetscFEGetBasisSpace(fe, &sp));
    CHKERRQ(PetscSpaceGetDegree(sp, &deg, NULL));
    CHKERRQ(PetscFEGetQuadrature(fe, &q));
    CHKERRQ(PetscQuadratureGetOrder(q, &ord));
    CHKERRQ_CEED(CeedBasisCreateTensorH1Lagrange(fe->ceed, dim, Nc, deg+1, (ord+1)/2, CEED_GAUSS, &fe->ceedBasis));
  }
  *basis = fe->ceedBasis;
  PetscFunctionReturn(0);
}

#endif
