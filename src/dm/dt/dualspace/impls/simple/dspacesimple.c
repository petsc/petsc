#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petscdmplex.h>

static PetscErrorCode PetscDualSpaceSetUp_Simple(PetscDualSpace sp)
{
  PetscDualSpace_Simple *s  = (PetscDualSpace_Simple *) sp->data;
  DM                     dm = sp->dm;
  PetscInt               dim;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = DMGetDimension(dm, &dim);CHKERRQ(ierr);
  ierr = PetscCalloc1(dim+1, &s->numDof);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceDestroy_Simple(PetscDualSpace sp)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  ierr = PetscFree(s->numDof);CHKERRQ(ierr);
  ierr = PetscFree(s);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceSimpleSetDimension_C", NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceSimpleSetFunctional_C", NULL);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceDuplicate_Simple(PetscDualSpace sp, PetscDualSpace *spNew)
{
  PetscInt       dim, d, Nc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscDualSpaceCreate(PetscObjectComm((PetscObject) sp), spNew);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetType(*spNew, PETSCDUALSPACESIMPLE);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetNumComponents(sp, &Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceSetNumComponents(sp, Nc);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDimension(sp, &dim);CHKERRQ(ierr);
  ierr = PetscDualSpaceSimpleSetDimension(*spNew, dim);CHKERRQ(ierr);
  for (d = 0; d < dim; ++d) {
    PetscQuadrature q;

    ierr = PetscDualSpaceGetFunctional(sp, d, &q);CHKERRQ(ierr);
    ierr = PetscDualSpaceSimpleSetFunctional(*spNew, d, q);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSetFromOptions_Simple(PetscOptionItems *PetscOptionsObject,PetscDualSpace sp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceGetDimension_Simple(PetscDualSpace sp, PetscInt *dim)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;

  PetscFunctionBegin;
  *dim = s->dim;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSimpleSetDimension_Simple(PetscDualSpace sp, const PetscInt dim)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;
  DM                     dm;
  PetscInt               spatialDim, f;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  for (f = 0; f < s->dim; ++f) {ierr = PetscQuadratureDestroy(&sp->functional[f]);CHKERRQ(ierr);}
  ierr = PetscFree(sp->functional);CHKERRQ(ierr);
  s->dim = dim;
  ierr = PetscCalloc1(s->dim, &sp->functional);CHKERRQ(ierr);
  ierr = PetscFree(s->numDof);CHKERRQ(ierr);
  ierr = PetscDualSpaceGetDM(sp, &dm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &spatialDim);CHKERRQ(ierr);
  ierr = PetscCalloc1(spatialDim+1, &s->numDof);CHKERRQ(ierr);
  s->numDof[spatialDim] = dim;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceGetNumDof_Simple(PetscDualSpace sp, const PetscInt **numDof)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;

  PetscFunctionBegin;
  *numDof = s->numDof;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSimpleSetFunctional_Simple(PetscDualSpace sp, PetscInt f, PetscQuadrature q)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;
  PetscReal             *weights;
  PetscInt               Nc, c, Nq, p;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  if ((f < 0) || (f >= s->dim)) SETERRQ2(PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_OUTOFRANGE, "Basis index %d not in [0, %d)", f, s->dim);
  ierr = PetscQuadratureDuplicate(q, &sp->functional[f]);CHKERRQ(ierr);
  /* Reweight so that it has unit volume: Do we want to do this for Nc > 1? */
  ierr = PetscQuadratureGetData(sp->functional[f], NULL, &Nc, &Nq, NULL, (const PetscReal **) &weights);CHKERRQ(ierr);
  for (c = 0; c < Nc; ++c) {
    PetscReal vol = 0.0;

    for (p = 0; p < Nq; ++p) vol += weights[p*Nc+c];
    for (p = 0; p < Nq; ++p) weights[p*Nc+c] /= (vol == 0.0 ? 1.0 : vol);
  }
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceSimpleSetDimension - Set the number of functionals in the dual space basis

  Logically Collective on sp

  Input Parameters:
+ sp  - the PetscDualSpace
- dim - the basis dimension

  Level: intermediate

.seealso: PetscDualSpaceSimpleSetFunctional()
@*/
PetscErrorCode PetscDualSpaceSimpleSetDimension(PetscDualSpace sp, PetscInt dim)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidLogicalCollectiveInt(sp, dim, 2);
  ierr = PetscTryMethod(sp, "PetscDualSpaceSimpleSetDimension_C", (PetscDualSpace,PetscInt),(sp,dim));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscDualSpaceSimpleSetFunctional - Set the given basis element for this dual space

  Not Collective

  Input Parameters:
+ sp  - the PetscDualSpace
. f - the basis index
- q - the basis functional

  Level: intermediate

  Note: The quadrature will be reweighted so that it has unit volume.

.seealso: PetscDualSpaceSimpleSetDimension()
@*/
PetscErrorCode PetscDualSpaceSimpleSetFunctional(PetscDualSpace sp, PetscInt func, PetscQuadrature q)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr = PetscTryMethod(sp, "PetscDualSpaceSimpleSetFunctional_C", (PetscDualSpace,PetscInt,PetscQuadrature),(sp,func,q));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceInitialize_Simple(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions    = PetscDualSpaceSetFromOptions_Simple;
  sp->ops->setup             = PetscDualSpaceSetUp_Simple;
  sp->ops->view              = NULL;
  sp->ops->destroy           = PetscDualSpaceDestroy_Simple;
  sp->ops->duplicate         = PetscDualSpaceDuplicate_Simple;
  sp->ops->getdimension      = PetscDualSpaceGetDimension_Simple;
  sp->ops->getnumdof         = PetscDualSpaceGetNumDof_Simple;
  sp->ops->getheightsubspace = NULL;
  sp->ops->getsymmetries     = NULL;
  sp->ops->apply             = PetscDualSpaceApplyDefault;
  sp->ops->applyall          = PetscDualSpaceApplyAllDefault;
  sp->ops->createallpoints   = PetscDualSpaceCreateAllPointsDefault;
  PetscFunctionReturn(0);
}

/*MC
  PETSCDUALSPACESIMPLE = "simple" - A PetscDualSpace object that encapsulates a dual space of arbitrary functionals

  Level: intermediate

.seealso: PetscDualSpaceType, PetscDualSpaceCreate(), PetscDualSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscDualSpaceCreate_Simple(PetscDualSpace sp)
{
  PetscDualSpace_Simple *s;
  PetscErrorCode         ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&s);CHKERRQ(ierr);
  sp->data = s;

  s->dim    = 0;
  s->numDof = NULL;

  ierr = PetscDualSpaceInitialize_Simple(sp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceSimpleSetDimension_C", PetscDualSpaceSimpleSetDimension_Simple);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceSimpleSetFunctional_C", PetscDualSpaceSimpleSetFunctional_Simple);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
