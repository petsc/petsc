#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petscdmplex.h>

static PetscErrorCode PetscDualSpaceSetUp_Simple(PetscDualSpace sp)
{
  PetscDualSpace_Simple *s  = (PetscDualSpace_Simple *) sp->data;
  DM                     dm = sp->dm;
  PetscInt               dim, pStart, pEnd;
  PetscSection           section;

  PetscFunctionBegin;
  CHKERRQ(DMGetDimension(dm, &dim));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  CHKERRQ(PetscSectionCreate(PETSC_COMM_SELF, &section));
  CHKERRQ(PetscSectionSetChart(section, pStart, pEnd));
  CHKERRQ(PetscSectionSetDof(section, pStart, s->dim));
  CHKERRQ(PetscSectionSetUp(section));
  sp->pointSection = section;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceDestroy_Simple(PetscDualSpace sp)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(s->numDof));
  CHKERRQ(PetscFree(s));
  CHKERRQ(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceSimpleSetDimension_C", NULL));
  CHKERRQ(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceSimpleSetFunctional_C", NULL));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceDuplicate_Simple(PetscDualSpace sp, PetscDualSpace spNew)
{
  PetscInt       dim, d;

  PetscFunctionBegin;
  CHKERRQ(PetscDualSpaceGetDimension(sp, &dim));
  CHKERRQ(PetscDualSpaceSimpleSetDimension(spNew, dim));
  for (d = 0; d < dim; ++d) {
    PetscQuadrature q;

    CHKERRQ(PetscDualSpaceGetFunctional(sp, d, &q));
    CHKERRQ(PetscDualSpaceSimpleSetFunctional(spNew, d, q));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSetFromOptions_Simple(PetscOptionItems *PetscOptionsObject,PetscDualSpace sp)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSimpleSetDimension_Simple(PetscDualSpace sp, const PetscInt dim)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;
  DM                     dm;
  PetscInt               spatialDim, f;

  PetscFunctionBegin;
  for (f = 0; f < s->dim; ++f) CHKERRQ(PetscQuadratureDestroy(&sp->functional[f]));
  CHKERRQ(PetscFree(sp->functional));
  s->dim = dim;
  CHKERRQ(PetscCalloc1(s->dim, &sp->functional));
  CHKERRQ(PetscFree(s->numDof));
  CHKERRQ(PetscDualSpaceGetDM(sp, &dm));
  CHKERRQ(DMGetCoordinateDim(dm, &spatialDim));
  CHKERRQ(PetscCalloc1(spatialDim+1, &s->numDof));
  s->numDof[spatialDim] = dim;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceSimpleSetFunctional_Simple(PetscDualSpace sp, PetscInt f, PetscQuadrature q)
{
  PetscDualSpace_Simple *s = (PetscDualSpace_Simple *) sp->data;
  PetscReal             *weights;
  PetscInt               Nc, c, Nq, p;

  PetscFunctionBegin;
  PetscCheckFalse((f < 0) || (f >= s->dim),PetscObjectComm((PetscObject) sp), PETSC_ERR_ARG_OUTOFRANGE, "Basis index %d not in [0, %d)", f, s->dim);
  CHKERRQ(PetscQuadratureDuplicate(q, &sp->functional[f]));
  /* Reweight so that it has unit volume: Do we want to do this for Nc > 1? */
  CHKERRQ(PetscQuadratureGetData(sp->functional[f], NULL, &Nc, &Nq, NULL, (const PetscReal **) &weights));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  PetscValidLogicalCollectiveInt(sp, dim, 2);
  PetscCheckFalse(sp->setupcalled,PetscObjectComm((PetscObject)sp), PETSC_ERR_ARG_WRONGSTATE, "Cannot change dimension after dual space is set up");
  CHKERRQ(PetscTryMethod(sp, "PetscDualSpaceSimpleSetDimension_C", (PetscDualSpace,PetscInt),(sp,dim)));
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
  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  CHKERRQ(PetscTryMethod(sp, "PetscDualSpaceSimpleSetFunctional_C", (PetscDualSpace,PetscInt,PetscQuadrature),(sp,func,q)));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscDualSpaceInitialize_Simple(PetscDualSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions       = PetscDualSpaceSetFromOptions_Simple;
  sp->ops->setup                = PetscDualSpaceSetUp_Simple;
  sp->ops->view                 = NULL;
  sp->ops->destroy              = PetscDualSpaceDestroy_Simple;
  sp->ops->duplicate            = PetscDualSpaceDuplicate_Simple;
  sp->ops->createheightsubspace = NULL;
  sp->ops->createpointsubspace  = NULL;
  sp->ops->getsymmetries        = NULL;
  sp->ops->apply                = PetscDualSpaceApplyDefault;
  sp->ops->applyall             = PetscDualSpaceApplyAllDefault;
  sp->ops->applyint             = PetscDualSpaceApplyInteriorDefault;
  sp->ops->createalldata        = PetscDualSpaceCreateAllDataDefault;
  sp->ops->createintdata        = PetscDualSpaceCreateInteriorDataDefault;
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCDUALSPACE_CLASSID, 1);
  CHKERRQ(PetscNewLog(sp,&s));
  sp->data = s;

  s->dim    = 0;
  s->numDof = NULL;

  CHKERRQ(PetscDualSpaceInitialize_Simple(sp));
  CHKERRQ(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceSimpleSetDimension_C", PetscDualSpaceSimpleSetDimension_Simple));
  CHKERRQ(PetscObjectComposeFunction((PetscObject) sp, "PetscDualSpaceSimpleSetFunctional_C", PetscDualSpaceSimpleSetFunctional_Simple));
  PetscFunctionReturn(0);
}
