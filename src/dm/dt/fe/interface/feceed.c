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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe, PETSCFE_CLASSID, 1);
  if (fe->ceed == ceed) PetscFunctionReturn(0);
  ierr = CeedReferenceCopy(ceed, &fe->ceed);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(fe, PETSCFE_CLASSID, 1);
  PetscValidBoolPointer(basis, 2);
  if (!fe->ceedBasis && fe->ceed) {
    ierr = PetscFEGetSpatialDimension(fe, &dim);CHKERRQ(ierr);
    ierr = PetscFEGetNumComponents(fe, &Nc);CHKERRQ(ierr);
    ierr = PetscFEGetBasisSpace(fe, &sp);CHKERRQ(ierr);
    ierr = PetscSpaceGetDegree(sp, &deg, NULL);CHKERRQ(ierr);
    ierr = PetscFEGetQuadrature(fe, &q);CHKERRQ(ierr);
    ierr = PetscQuadratureGetOrder(q, &ord);CHKERRQ(ierr);
    ierr = CeedBasisCreateTensorH1Lagrange(fe->ceed, dim, Nc, deg+1, (ord+1)/2, CEED_GAUSS, &fe->ceedBasis);CHKERRQ(ierr);
  }
  *basis = fe->ceedBasis;
  PetscFunctionReturn(0);
}

#endif
