#include <petscdmadaptor.h>            /*I "petscdmadaptor.h" I*/
#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscds.h>
#include <petscblaslapack.h>

#include <petsc/private/dmadaptorimpl.h>
#include <petsc/private/dmpleximpl.h>
#include <petsc/private/petscfeimpl.h>

static PetscErrorCode DMAdaptorSimpleErrorIndicator_Private(DMAdaptor, PetscInt, PetscInt, const PetscScalar *, const PetscScalar *, const PetscFVCellGeom *, PetscReal *, void *);

static PetscErrorCode DMAdaptorTransferSolution_Exact_Private(DMAdaptor adaptor, DM dm, Vec u, DM adm, Vec au, void *ctx)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMProjectFunction(adm, 0.0, adaptor->exactSol, adaptor->exactCtx, INSERT_ALL_VALUES, au));
  PetscFunctionReturn(0);
}

/*@
  DMAdaptorCreate - Create a DMAdaptor object. Its purpose is to construct a adaptation DMLabel or metric Vec that can be used to modify the DM.

  Collective

  Input Parameter:
. comm - The communicator for the DMAdaptor object

  Output Parameter:
. adaptor   - The DMAdaptor object

  Level: beginner

.seealso: DMAdaptorDestroy(), DMAdaptorAdapt()
@*/
PetscErrorCode DMAdaptorCreate(MPI_Comm comm, DMAdaptor *adaptor)
{
  VecTaggerBox     refineBox, coarsenBox;

  PetscFunctionBegin;
  PetscValidPointer(adaptor, 2);
  CHKERRQ(PetscSysInitializePackage());
  CHKERRQ(PetscHeaderCreate(*adaptor, DM_CLASSID, "DMAdaptor", "DM Adaptor", "SNES", comm, DMAdaptorDestroy, DMAdaptorView));

  (*adaptor)->monitor = PETSC_FALSE;
  (*adaptor)->adaptCriterion = DM_ADAPTATION_NONE;
  (*adaptor)->numSeq = 1;
  (*adaptor)->Nadapt = -1;
  (*adaptor)->refinementFactor = 2.0;
  (*adaptor)->ops->computeerrorindicator = DMAdaptorSimpleErrorIndicator_Private;
  refineBox.min = refineBox.max = PETSC_MAX_REAL;
  CHKERRQ(VecTaggerCreate(PetscObjectComm((PetscObject) *adaptor), &(*adaptor)->refineTag));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) (*adaptor)->refineTag, "refine_"));
  CHKERRQ(VecTaggerSetType((*adaptor)->refineTag, VECTAGGERABSOLUTE));
  CHKERRQ(VecTaggerAbsoluteSetBox((*adaptor)->refineTag, &refineBox));
  coarsenBox.min = coarsenBox.max = PETSC_MAX_REAL;
  CHKERRQ(VecTaggerCreate(PetscObjectComm((PetscObject) *adaptor), &(*adaptor)->coarsenTag));
  CHKERRQ(PetscObjectSetOptionsPrefix((PetscObject) (*adaptor)->coarsenTag, "coarsen_"));
  CHKERRQ(VecTaggerSetType((*adaptor)->coarsenTag, VECTAGGERABSOLUTE));
  CHKERRQ(VecTaggerAbsoluteSetBox((*adaptor)->coarsenTag, &coarsenBox));
  PetscFunctionReturn(0);
}

/*@
  DMAdaptorDestroy - Destroys a DMAdaptor object

  Collective on DMAdaptor

  Input Parameter:
. adaptor - The DMAdaptor object

  Level: beginner

.seealso: DMAdaptorCreate(), DMAdaptorAdapt()
@*/
PetscErrorCode DMAdaptorDestroy(DMAdaptor *adaptor)
{
  PetscFunctionBegin;
  if (!*adaptor) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*adaptor), DM_CLASSID, 1);
  if (--((PetscObject)(*adaptor))->refct > 0) {
    *adaptor = NULL;
    PetscFunctionReturn(0);
  }
  CHKERRQ(VecTaggerDestroy(&(*adaptor)->refineTag));
  CHKERRQ(VecTaggerDestroy(&(*adaptor)->coarsenTag));
  CHKERRQ(PetscFree2((*adaptor)->exactSol, (*adaptor)->exactCtx));
  CHKERRQ(PetscHeaderDestroy(adaptor));
  PetscFunctionReturn(0);
}

/*@
  DMAdaptorSetFromOptions - Sets a DMAdaptor object from options

  Collective on DMAdaptor

  Input Parameter:
. adaptor - The DMAdaptor object

  Options Database Keys:
+ -adaptor_monitor <bool>        - Monitor the adaptation process
. -adaptor_sequence_num <num>    - Number of adaptations to generate an optimal grid
. -adaptor_target_num <num>      - Set the target number of vertices N_adapt, -1 for automatic determination
- -adaptor_refinement_factor <r> - Set r such that N_adapt = r^dim N_orig

  Level: beginner

.seealso: DMAdaptorCreate(), DMAdaptorAdapt()
@*/
PetscErrorCode DMAdaptorSetFromOptions(DMAdaptor adaptor)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject) adaptor), "", "DM Adaptor Options", "DMAdaptor");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-adaptor_monitor", "Monitor the adaptation process", "DMAdaptorMonitor", adaptor->monitor, &adaptor->monitor, NULL));
  CHKERRQ(PetscOptionsInt("-adaptor_sequence_num", "Number of adaptations to generate an optimal grid", "DMAdaptorSetSequenceLength", adaptor->numSeq, &adaptor->numSeq, NULL));
  CHKERRQ(PetscOptionsInt("-adaptor_target_num", "Set the target number of vertices N_adapt, -1 for automatic determination", "DMAdaptor", adaptor->Nadapt, &adaptor->Nadapt, NULL));
  CHKERRQ(PetscOptionsReal("-adaptor_refinement_factor", "Set r such that N_adapt = r^dim N_orig", "DMAdaptor", adaptor->refinementFactor, &adaptor->refinementFactor, NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);
  CHKERRQ(VecTaggerSetFromOptions(adaptor->refineTag));
  CHKERRQ(VecTaggerSetFromOptions(adaptor->coarsenTag));
  PetscFunctionReturn(0);
}

/*@
  DMAdaptorView - Views a DMAdaptor object

  Collective on DMAdaptor

  Input Parameters:
+ adaptor     - The DMAdaptor object
- viewer - The PetscViewer object

  Level: beginner

.seealso: DMAdaptorCreate(), DMAdaptorAdapt()
@*/
PetscErrorCode DMAdaptorView(DMAdaptor adaptor, PetscViewer viewer)
{
  PetscFunctionBegin;
  CHKERRQ(PetscObjectPrintClassNamePrefixType((PetscObject) adaptor, viewer));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "DM Adaptor\n"));
  CHKERRQ(PetscViewerASCIIPrintf(viewer, "  sequence length: %D\n", adaptor->numSeq));
  CHKERRQ(VecTaggerView(adaptor->refineTag,  viewer));
  CHKERRQ(VecTaggerView(adaptor->coarsenTag, viewer));
  PetscFunctionReturn(0);
}

/*@
  DMAdaptorGetSolver - Gets the solver used to produce discrete solutions

  Not collective

  Input Parameter:
. adaptor   - The DMAdaptor object

  Output Parameter:
. snes - The solver

  Level: intermediate

.seealso: DMAdaptorSetSolver(), DMAdaptorCreate(), DMAdaptorAdapt()
@*/
PetscErrorCode DMAdaptorGetSolver(DMAdaptor adaptor, SNES *snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DM_CLASSID, 1);
  PetscValidPointer(snes, 2);
  *snes = adaptor->snes;
  PetscFunctionReturn(0);
}

/*@
  DMAdaptorSetSolver - Sets the solver used to produce discrete solutions

  Not collective

  Input Parameters:
+ adaptor   - The DMAdaptor object
- snes - The solver

  Level: intermediate

  Note: The solver MUST have an attached DM/DS, so that we know the exact solution

.seealso: DMAdaptorGetSolver(), DMAdaptorCreate(), DMAdaptorAdapt()
@*/
PetscErrorCode DMAdaptorSetSolver(DMAdaptor adaptor, SNES snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DM_CLASSID, 1);
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 2);
  adaptor->snes = snes;
  CHKERRQ(SNESGetDM(adaptor->snes, &adaptor->idm));
  PetscFunctionReturn(0);
}

/*@
  DMAdaptorGetSequenceLength - Gets the number of sequential adaptations

  Not collective

  Input Parameter:
. adaptor - The DMAdaptor object

  Output Parameter:
. num - The number of adaptations

  Level: intermediate

.seealso: DMAdaptorSetSequenceLength(), DMAdaptorCreate(), DMAdaptorAdapt()
@*/
PetscErrorCode DMAdaptorGetSequenceLength(DMAdaptor adaptor, PetscInt *num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DM_CLASSID, 1);
  PetscValidPointer(num, 2);
  *num = adaptor->numSeq;
  PetscFunctionReturn(0);
}

/*@
  DMAdaptorSetSequenceLength - Sets the number of sequential adaptations

  Not collective

  Input Parameters:
+ adaptor - The DMAdaptor object
- num - The number of adaptations

  Level: intermediate

.seealso: DMAdaptorGetSequenceLength(), DMAdaptorCreate(), DMAdaptorAdapt()
@*/
PetscErrorCode DMAdaptorSetSequenceLength(DMAdaptor adaptor, PetscInt num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(adaptor, DM_CLASSID, 1);
  adaptor->numSeq = num;
  PetscFunctionReturn(0);
}

/*@
  DMAdaptorSetUp - After the solver is specified, we create structures for controlling adaptivity

  Collective on DMAdaptor

  Input Parameters:
. adaptor - The DMAdaptor object

  Level: beginner

.seealso: DMAdaptorCreate(), DMAdaptorAdapt()
@*/
PetscErrorCode DMAdaptorSetUp(DMAdaptor adaptor)
{
  PetscDS        prob;
  PetscInt       Nf, f;

  PetscFunctionBegin;
  CHKERRQ(DMGetDS(adaptor->idm, &prob));
  CHKERRQ(VecTaggerSetUp(adaptor->refineTag));
  CHKERRQ(VecTaggerSetUp(adaptor->coarsenTag));
  CHKERRQ(PetscDSGetNumFields(prob, &Nf));
  CHKERRQ(PetscMalloc2(Nf, &adaptor->exactSol, Nf, &adaptor->exactCtx));
  for (f = 0; f < Nf; ++f) {
    CHKERRQ(PetscDSGetExactSolution(prob, f, &adaptor->exactSol[f], &adaptor->exactCtx[f]));
    /* TODO Have a flag that forces projection rather than using the exact solution */
    if (adaptor->exactSol[0]) CHKERRQ(DMAdaptorSetTransferFunction(adaptor, DMAdaptorTransferSolution_Exact_Private));
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMAdaptorGetTransferFunction(DMAdaptor adaptor, PetscErrorCode (**tfunc)(DMAdaptor, DM, Vec, DM, Vec, void *))
{
  PetscFunctionBegin;
  *tfunc = adaptor->ops->transfersolution;
  PetscFunctionReturn(0);
}

PetscErrorCode DMAdaptorSetTransferFunction(DMAdaptor adaptor, PetscErrorCode (*tfunc)(DMAdaptor, DM, Vec, DM, Vec, void *))
{
  PetscFunctionBegin;
  adaptor->ops->transfersolution = tfunc;
  PetscFunctionReturn(0);
}

PetscErrorCode DMAdaptorPreAdapt(DMAdaptor adaptor, Vec locX)
{
  DM             plex;
  PetscDS        prob;
  PetscObject    obj;
  PetscClassId   id;
  PetscBool      isForest;

  PetscFunctionBegin;
  CHKERRQ(DMConvert(adaptor->idm, DMPLEX, &plex));
  CHKERRQ(DMGetDS(adaptor->idm, &prob));
  CHKERRQ(PetscDSGetDiscretization(prob, 0, &obj));
  CHKERRQ(PetscObjectGetClassId(obj, &id));
  CHKERRQ(DMIsForest(adaptor->idm, &isForest));
  if (adaptor->adaptCriterion == DM_ADAPTATION_NONE) {
    if (isForest) {adaptor->adaptCriterion = DM_ADAPTATION_LABEL;}
#if defined(PETSC_HAVE_PRAGMATIC)
    else          {adaptor->adaptCriterion = DM_ADAPTATION_METRIC;}
#elif defined(PETSC_HAVE_MMG)
    else          {adaptor->adaptCriterion = DM_ADAPTATION_METRIC;}
#elif defined(PETSC_HAVE_PARMMG)
    else          {adaptor->adaptCriterion = DM_ADAPTATION_METRIC;}
#else
    else          {adaptor->adaptCriterion = DM_ADAPTATION_REFINE;}
#endif
  }
  if (id == PETSCFV_CLASSID) {adaptor->femType = PETSC_FALSE;}
  else                       {adaptor->femType = PETSC_TRUE;}
  if (adaptor->femType) {
    /* Compute local solution bc */
    CHKERRQ(DMPlexInsertBoundaryValues(plex, PETSC_TRUE, locX, 0.0, adaptor->faceGeom, adaptor->cellGeom, NULL));
  } else {
    PetscFV      fvm = (PetscFV) obj;
    PetscLimiter noneLimiter;
    Vec          grad;

    CHKERRQ(PetscFVGetComputeGradients(fvm, &adaptor->computeGradient));
    CHKERRQ(PetscFVSetComputeGradients(fvm, PETSC_TRUE));
    /* Use no limiting when reconstructing gradients for adaptivity */
    CHKERRQ(PetscFVGetLimiter(fvm, &adaptor->limiter));
    CHKERRQ(PetscObjectReference((PetscObject) adaptor->limiter));
    CHKERRQ(PetscLimiterCreate(PetscObjectComm((PetscObject) fvm), &noneLimiter));
    CHKERRQ(PetscLimiterSetType(noneLimiter, PETSCLIMITERNONE));
    CHKERRQ(PetscFVSetLimiter(fvm, noneLimiter));
    /* Get FVM data */
    CHKERRQ(DMPlexGetDataFVM(plex, fvm, &adaptor->cellGeom, &adaptor->faceGeom, &adaptor->gradDM));
    CHKERRQ(VecGetDM(adaptor->cellGeom, &adaptor->cellDM));
    CHKERRQ(VecGetArrayRead(adaptor->cellGeom, &adaptor->cellGeomArray));
    /* Compute local solution bc */
    CHKERRQ(DMPlexInsertBoundaryValues(plex, PETSC_TRUE, locX, 0.0, adaptor->faceGeom, adaptor->cellGeom, NULL));
    /* Compute gradients */
    CHKERRQ(DMCreateGlobalVector(adaptor->gradDM, &grad));
    CHKERRQ(DMPlexReconstructGradientsFVM(plex, locX, grad));
    CHKERRQ(DMGetLocalVector(adaptor->gradDM, &adaptor->cellGrad));
    CHKERRQ(DMGlobalToLocalBegin(adaptor->gradDM, grad, INSERT_VALUES, adaptor->cellGrad));
    CHKERRQ(DMGlobalToLocalEnd(adaptor->gradDM, grad, INSERT_VALUES, adaptor->cellGrad));
    CHKERRQ(VecDestroy(&grad));
    CHKERRQ(VecGetArrayRead(adaptor->cellGrad, &adaptor->cellGradArray));
  }
  CHKERRQ(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

PetscErrorCode DMAdaptorTransferSolution(DMAdaptor adaptor, DM dm, Vec x, DM adm, Vec ax)
{
  PetscReal      time = 0.0;
  Mat            interp;
  void          *ctx;

  PetscFunctionBegin;
  CHKERRQ(DMGetApplicationContext(dm, &ctx));
  if (adaptor->ops->transfersolution) {
    CHKERRQ((*adaptor->ops->transfersolution)(adaptor, dm, x, adm, ax, ctx));
  } else {
    switch (adaptor->adaptCriterion) {
    case DM_ADAPTATION_LABEL:
      CHKERRQ(DMForestTransferVec(dm, x, adm, ax, PETSC_TRUE, time));
      break;
    case DM_ADAPTATION_REFINE:
    case DM_ADAPTATION_METRIC:
      CHKERRQ(DMCreateInterpolation(dm, adm, &interp, NULL));
      CHKERRQ(MatInterpolate(interp, x, ax));
      CHKERRQ(DMInterpolate(dm, interp, adm));
      CHKERRQ(MatDestroy(&interp));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject) adaptor), PETSC_ERR_SUP, "No built-in projection for this adaptation criterion: %D", adaptor->adaptCriterion);
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMAdaptorPostAdapt(DMAdaptor adaptor)
{
  PetscDS        prob;
  PetscObject    obj;
  PetscClassId   id;

  PetscFunctionBegin;
  CHKERRQ(DMGetDS(adaptor->idm, &prob));
  CHKERRQ(PetscDSGetDiscretization(prob, 0, &obj));
  CHKERRQ(PetscObjectGetClassId(obj, &id));
  if (id == PETSCFV_CLASSID) {
    PetscFV fvm = (PetscFV) obj;

    CHKERRQ(PetscFVSetComputeGradients(fvm, adaptor->computeGradient));
    /* Restore original limiter */
    CHKERRQ(PetscFVSetLimiter(fvm, adaptor->limiter));

    CHKERRQ(VecRestoreArrayRead(adaptor->cellGeom, &adaptor->cellGeomArray));
    CHKERRQ(VecRestoreArrayRead(adaptor->cellGrad, &adaptor->cellGradArray));
    CHKERRQ(DMRestoreLocalVector(adaptor->gradDM, &adaptor->cellGrad));
  }
  PetscFunctionReturn(0);
}

/*
  DMAdaptorSimpleErrorIndicator - Just use the integrated gradient as an error indicator

  Input Parameters:
+ adaptor  - The DMAdaptor object
. dim      - The topological dimension
. cell     - The cell
. field    - The field integrated over the cell
. gradient - The gradient integrated over the cell
. cg       - A PetscFVCellGeom struct
- ctx      - A user context

  Output Parameter:
. errInd   - The error indicator

.seealso: DMAdaptorComputeErrorIndicator()
*/
static PetscErrorCode DMAdaptorSimpleErrorIndicator_Private(DMAdaptor adaptor, PetscInt dim, PetscInt Nc, const PetscScalar *field, const PetscScalar *gradient, const PetscFVCellGeom *cg, PetscReal *errInd, void *ctx)
{
  PetscReal err = 0.;
  PetscInt  c, d;

  PetscFunctionBeginHot;
  for (c = 0; c < Nc; c++) {
    for (d = 0; d < dim; ++d) {
      err += PetscSqr(PetscRealPart(gradient[c*dim+d]));
    }
  }
  *errInd = cg->volume * err;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMAdaptorComputeErrorIndicator_Private(DMAdaptor adaptor, DM plex, PetscInt cell, Vec locX, PetscReal *errInd)
{
  PetscDS         prob;
  PetscObject     obj;
  PetscClassId    id;
  void           *ctx;
  PetscQuadrature quad;
  PetscInt        dim, d, cdim, Nc;

  PetscFunctionBegin;
  *errInd = 0.;
  CHKERRQ(DMGetDimension(plex, &dim));
  CHKERRQ(DMGetCoordinateDim(plex, &cdim));
  CHKERRQ(DMGetApplicationContext(plex, &ctx));
  CHKERRQ(DMGetDS(plex, &prob));
  CHKERRQ(PetscDSGetDiscretization(prob, 0, &obj));
  CHKERRQ(PetscObjectGetClassId(obj, &id));
  if (id == PETSCFV_CLASSID) {
    const PetscScalar *pointSols;
    const PetscScalar *pointSol;
    const PetscScalar *pointGrad;
    PetscFVCellGeom   *cg;

    CHKERRQ(PetscFVGetNumComponents((PetscFV) obj, &Nc));
    CHKERRQ(VecGetArrayRead(locX, &pointSols));
    CHKERRQ(DMPlexPointLocalRead(plex, cell, pointSols, (void *) &pointSol));
    CHKERRQ(DMPlexPointLocalRead(adaptor->gradDM, cell, adaptor->cellGradArray, (void *) &pointGrad));
    CHKERRQ(DMPlexPointLocalRead(adaptor->cellDM, cell, adaptor->cellGeomArray, &cg));
    CHKERRQ((*adaptor->ops->computeerrorindicator)(adaptor, dim, Nc, pointSol, pointGrad, cg, errInd, ctx));
    CHKERRQ(VecRestoreArrayRead(locX, &pointSols));
  } else {
    PetscScalar     *x = NULL, *field, *gradient, *interpolant, *interpolantGrad;
    PetscFVCellGeom  cg;
    PetscFEGeom      fegeom;
    const PetscReal *quadWeights;
    PetscReal       *coords;
    PetscInt         Nb, fc, Nq, qNc, Nf, f, fieldOffset;

    fegeom.dim      = dim;
    fegeom.dimEmbed = cdim;
    CHKERRQ(PetscDSGetNumFields(prob, &Nf));
    CHKERRQ(PetscFEGetQuadrature((PetscFE) obj, &quad));
    CHKERRQ(DMPlexVecGetClosure(plex, NULL, locX, cell, NULL, &x));
    CHKERRQ(PetscFEGetDimension((PetscFE) obj, &Nb));
    CHKERRQ(PetscFEGetNumComponents((PetscFE) obj, &Nc));
    CHKERRQ(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, NULL, &quadWeights));
    CHKERRQ(PetscMalloc6(Nc,&field,cdim*Nc,&gradient,cdim*Nq,&coords,Nq,&fegeom.detJ,cdim*cdim*Nq,&fegeom.J,cdim*cdim*Nq,&fegeom.invJ));
    CHKERRQ(PetscMalloc2(Nc, &interpolant, cdim*Nc, &interpolantGrad));
    CHKERRQ(DMPlexComputeCellGeometryFEM(plex, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
    CHKERRQ(DMPlexComputeCellGeometryFVM(plex, cell, &cg.volume, NULL, NULL));
    CHKERRQ(PetscArrayzero(gradient, cdim*Nc));
    for (f = 0, fieldOffset = 0; f < Nf; ++f) {
      PetscInt qc = 0, q;

      CHKERRQ(PetscDSGetDiscretization(prob, f, &obj));
      CHKERRQ(PetscArrayzero(interpolant,Nc));
      CHKERRQ(PetscArrayzero(interpolantGrad, cdim*Nc));
      for (q = 0; q < Nq; ++q) {
        CHKERRQ(PetscFEInterpolateFieldAndGradient_Static((PetscFE) obj, 1, x, &fegeom, q, interpolant, interpolantGrad));
        for (fc = 0; fc < Nc; ++fc) {
          const PetscReal wt = quadWeights[q*qNc+qc+fc];

          field[fc] += interpolant[fc]*wt*fegeom.detJ[q];
          for (d = 0; d < cdim; ++d) gradient[fc*cdim+d] += interpolantGrad[fc*dim+d]*wt*fegeom.detJ[q];
        }
      }
      fieldOffset += Nb;
      qc          += Nc;
    }
    CHKERRQ(PetscFree2(interpolant, interpolantGrad));
    CHKERRQ(DMPlexVecRestoreClosure(plex, NULL, locX, cell, NULL, &x));
    for (fc = 0; fc < Nc; ++fc) {
      field[fc] /= cg.volume;
      for (d = 0; d < cdim; ++d) gradient[fc*cdim+d] /= cg.volume;
    }
    CHKERRQ((*adaptor->ops->computeerrorindicator)(adaptor, dim, Nc, field, gradient, &cg, errInd, ctx));
    CHKERRQ(PetscFree6(field,gradient,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
  }
  PetscFunctionReturn(0);
}

static void identityFunc(PetscInt dim, PetscInt Nf, PetscInt NfAux,
                         const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[],
                         const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[],
                         PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[])
{
  PetscInt i, j;

  for (i = 0; i < dim; ++i) {
    for (j = 0; j < dim; ++j) {
      f[i+dim*j] = u[i+dim*j];
    }
  }
}

static PetscErrorCode DMAdaptorAdapt_Sequence_Private(DMAdaptor adaptor, Vec inx, PetscBool doSolve, DM *adm, Vec *ax)
{
  PetscDS        prob;
  void          *ctx;
  MPI_Comm       comm;
  PetscInt       numAdapt = adaptor->numSeq, adaptIter;
  PetscInt       dim, coordDim, numFields, cStart, cEnd, c;

  PetscFunctionBegin;
  CHKERRQ(DMViewFromOptions(adaptor->idm, NULL, "-dm_adapt_pre_view"));
  CHKERRQ(VecViewFromOptions(inx, NULL, "-sol_adapt_pre_view"));
  CHKERRQ(PetscObjectGetComm((PetscObject) adaptor, &comm));
  CHKERRQ(DMGetDimension(adaptor->idm, &dim));
  CHKERRQ(DMGetCoordinateDim(adaptor->idm, &coordDim));
  CHKERRQ(DMGetApplicationContext(adaptor->idm, &ctx));
  CHKERRQ(DMGetDS(adaptor->idm, &prob));
  CHKERRQ(PetscDSGetNumFields(prob, &numFields));
  PetscCheckFalse(numFields == 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fields is zero!");

  /* Adapt until nothing changes */
  /* Adapt for a specified number of iterates */
  for (adaptIter = 0; adaptIter < numAdapt-1; ++adaptIter) CHKERRQ(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_(comm)));
  for (adaptIter = 0; adaptIter < numAdapt;   ++adaptIter) {
    PetscBool adapted = PETSC_FALSE;
    DM        dm      = adaptIter ? *adm : adaptor->idm, odm;
    Vec       x       = adaptIter ? *ax  : inx, locX, ox;

    CHKERRQ(DMGetLocalVector(dm, &locX));
    CHKERRQ(DMGlobalToLocalBegin(dm, adaptIter ? *ax : x, INSERT_VALUES, locX));
    CHKERRQ(DMGlobalToLocalEnd(dm, adaptIter ? *ax : x, INSERT_VALUES, locX));
    CHKERRQ(DMAdaptorPreAdapt(adaptor, locX));
    if (doSolve) {
      SNES snes;

      CHKERRQ(DMAdaptorGetSolver(adaptor, &snes));
      CHKERRQ(SNESSolve(snes, NULL, adaptIter ? *ax : x));
    }
    /* CHKERRQ(DMAdaptorMonitor(adaptor));
       Print iterate, memory used, DM, solution */
    switch (adaptor->adaptCriterion) {
    case DM_ADAPTATION_REFINE:
      CHKERRQ(DMRefine(dm, comm, &odm));
      PetscCheckFalse(!odm,comm, PETSC_ERR_ARG_INCOMP, "DMRefine() did not perform any refinement, cannot continue grid sequencing");
      adapted = PETSC_TRUE;
      break;
    case DM_ADAPTATION_LABEL:
    {
      /* Adapt DM
           Create local solution
           Reconstruct gradients (FVM) or solve adjoint equation (FEM)
           Produce cellwise error indicator */
      DM                 plex;
      DMLabel            adaptLabel;
      IS                 refineIS, coarsenIS;
      Vec                errVec;
      PetscScalar       *errArray;
      const PetscScalar *pointSols;
      PetscReal          minMaxInd[2] = {PETSC_MAX_REAL, PETSC_MIN_REAL}, minMaxIndGlobal[2];
      PetscInt           nRefine, nCoarsen;

      CHKERRQ(DMConvert(dm, DMPLEX, &plex));
      CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel));
      CHKERRQ(DMPlexGetSimplexOrBoxCells(plex, 0, &cStart, &cEnd));

      CHKERRQ(VecCreateMPI(PetscObjectComm((PetscObject) adaptor), cEnd-cStart, PETSC_DETERMINE, &errVec));
      CHKERRQ(VecSetUp(errVec));
      CHKERRQ(VecGetArray(errVec, &errArray));
      CHKERRQ(VecGetArrayRead(locX, &pointSols));
      for (c = cStart; c < cEnd; ++c) {
        PetscReal errInd;

        CHKERRQ(DMAdaptorComputeErrorIndicator_Private(adaptor, plex, c, locX, &errInd));
        errArray[c-cStart] = errInd;
        minMaxInd[0] = PetscMin(minMaxInd[0], errInd);
        minMaxInd[1] = PetscMax(minMaxInd[1], errInd);
      }
      CHKERRQ(VecRestoreArrayRead(locX, &pointSols));
      CHKERRQ(VecRestoreArray(errVec, &errArray));
      CHKERRQ(PetscGlobalMinMaxReal(PetscObjectComm((PetscObject) adaptor), minMaxInd, minMaxIndGlobal));
      CHKERRQ(PetscInfo(adaptor, "DMAdaptor: error indicator range (%E, %E)\n", minMaxIndGlobal[0], minMaxIndGlobal[1]));
      /*     Compute IS from VecTagger */
      CHKERRQ(VecTaggerComputeIS(adaptor->refineTag, errVec, &refineIS,NULL));
      CHKERRQ(VecTaggerComputeIS(adaptor->coarsenTag, errVec, &coarsenIS,NULL));
      CHKERRQ(ISGetSize(refineIS, &nRefine));
      CHKERRQ(ISGetSize(coarsenIS, &nCoarsen));
      CHKERRQ(PetscInfo(adaptor, "DMAdaptor: numRefine %D, numCoarsen %D\n", nRefine, nCoarsen));
      if (nRefine)  CHKERRQ(DMLabelSetStratumIS(adaptLabel, DM_ADAPT_REFINE,  refineIS));
      if (nCoarsen) CHKERRQ(DMLabelSetStratumIS(adaptLabel, DM_ADAPT_COARSEN, coarsenIS));
      CHKERRQ(ISDestroy(&coarsenIS));
      CHKERRQ(ISDestroy(&refineIS));
      CHKERRQ(VecDestroy(&errVec));
      /*     Adapt DM from label */
      if (nRefine || nCoarsen) {
        CHKERRQ(DMAdaptLabel(dm, adaptLabel, &odm));
        adapted = PETSC_TRUE;
      }
      CHKERRQ(DMLabelDestroy(&adaptLabel));
      CHKERRQ(DMDestroy(&plex));
    }
    break;
    case DM_ADAPTATION_METRIC:
    {
      DM           dmGrad, dmHess, dmMetric;
      Vec          xGrad, xHess, metric;
      PetscReal    N;
      DMLabel      bdLabel = NULL, rgLabel = NULL;
      PetscBool    higherOrder = PETSC_FALSE;
      PetscInt     Nd = coordDim*coordDim, f, vStart, vEnd;
      void       (**funcs)(PetscInt, PetscInt, PetscInt,
                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                           const PetscInt[], const PetscInt[], const PetscScalar[], const PetscScalar[], const PetscScalar[],
                           PetscReal, const PetscReal[], PetscInt, const PetscScalar[], PetscScalar[]);

      CHKERRQ(PetscMalloc(1, &funcs));
      funcs[0] = identityFunc;

      /*     Setup finite element spaces */
      CHKERRQ(DMClone(dm, &dmGrad));
      CHKERRQ(DMClone(dm, &dmHess));
      PetscCheckFalse(numFields > 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Adaptation with multiple fields not yet considered");  // TODO
      for (f = 0; f < numFields; ++f) {
        PetscFE         fe, feGrad, feHess;
        PetscDualSpace  Q;
        PetscSpace      space;
        DM              K;
        PetscQuadrature q;
        PetscInt        Nc, qorder, p;
        const char     *prefix;

        CHKERRQ(PetscDSGetDiscretization(prob, f, (PetscObject *) &fe));
        CHKERRQ(PetscFEGetNumComponents(fe, &Nc));
        PetscCheckFalse(Nc > 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Adaptation with multiple components not yet considered");  // TODO
        CHKERRQ(PetscFEGetBasisSpace(fe, &space));
        CHKERRQ(PetscSpaceGetDegree(space, NULL, &p));
        if (p > 1) higherOrder = PETSC_TRUE;
        CHKERRQ(PetscFEGetDualSpace(fe, &Q));
        CHKERRQ(PetscDualSpaceGetDM(Q, &K));
        CHKERRQ(DMPlexGetDepthStratum(K, 0, &vStart, &vEnd));
        CHKERRQ(PetscFEGetQuadrature(fe, &q));
        CHKERRQ(PetscQuadratureGetOrder(q, &qorder));
        CHKERRQ(PetscObjectGetOptionsPrefix((PetscObject) fe, &prefix));
        CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dmGrad), dim, Nc*coordDim, PETSC_TRUE, prefix, qorder, &feGrad));
        CHKERRQ(PetscFECreateDefault(PetscObjectComm((PetscObject) dmHess), dim, Nc*Nd, PETSC_TRUE, prefix, qorder, &feHess));
        CHKERRQ(DMSetField(dmGrad, f, NULL, (PetscObject)feGrad));
        CHKERRQ(DMSetField(dmHess, f, NULL, (PetscObject)feHess));
        CHKERRQ(DMCreateDS(dmGrad));
        CHKERRQ(DMCreateDS(dmHess));
        CHKERRQ(PetscFEDestroy(&feGrad));
        CHKERRQ(PetscFEDestroy(&feHess));
      }
      /*     Compute vertexwise gradients from cellwise gradients */
      CHKERRQ(DMCreateLocalVector(dmGrad, &xGrad));
      CHKERRQ(VecViewFromOptions(locX, NULL, "-sol_adapt_loc_pre_view"));
      CHKERRQ(DMPlexComputeGradientClementInterpolant(dm, locX, xGrad));
      CHKERRQ(VecViewFromOptions(xGrad, NULL, "-adapt_gradient_view"));
      /*     Compute vertexwise Hessians from cellwise Hessians */
      CHKERRQ(DMCreateLocalVector(dmHess, &xHess));
      CHKERRQ(DMPlexComputeGradientClementInterpolant(dmGrad, xGrad, xHess));
      CHKERRQ(VecViewFromOptions(xHess, NULL, "-adapt_hessian_view"));
      CHKERRQ(VecDestroy(&xGrad));
      CHKERRQ(DMDestroy(&dmGrad));
      /*     Compute L-p normalized metric */
      CHKERRQ(DMClone(dm, &dmMetric));
      N    = adaptor->Nadapt >= 0 ? adaptor->Nadapt : PetscPowRealInt(adaptor->refinementFactor, dim)*((PetscReal) (vEnd - vStart));
      if (adaptor->monitor) {
        PetscMPIInt rank, size;
        CHKERRMPI(MPI_Comm_rank(comm, &size));
        CHKERRMPI(MPI_Comm_rank(comm, &rank));
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "[%D] N_orig: %D N_adapt: %g\n", rank, vEnd - vStart, N));
      }
      CHKERRQ(DMPlexMetricSetTargetComplexity(dmMetric, (PetscReal) N));
      if (higherOrder) {
        /*   Project Hessian into P1 space, if required */
        CHKERRQ(DMPlexMetricCreate(dmMetric, 0, &metric));
        CHKERRQ(DMProjectFieldLocal(dmMetric, 0.0, xHess, funcs, INSERT_ALL_VALUES, metric));
        CHKERRQ(VecDestroy(&xHess));
        xHess = metric;
      }
      CHKERRQ(PetscFree(funcs));
      CHKERRQ(DMPlexMetricNormalize(dmMetric, xHess, PETSC_TRUE, PETSC_TRUE, &metric));
      CHKERRQ(VecDestroy(&xHess));
      CHKERRQ(DMDestroy(&dmHess));
      /*     Adapt DM from metric */
      CHKERRQ(DMGetLabel(dm, "marker", &bdLabel));
      CHKERRQ(DMAdaptMetric(dm, metric, bdLabel, rgLabel, &odm));
      adapted = PETSC_TRUE;
      /*     Cleanup */
      CHKERRQ(VecDestroy(&metric));
      CHKERRQ(DMDestroy(&dmMetric));
    }
    break;
    default: SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid adaptation type: %D", adaptor->adaptCriterion);
    }
    CHKERRQ(DMAdaptorPostAdapt(adaptor));
    CHKERRQ(DMRestoreLocalVector(dm, &locX));
    /* If DM was adapted, replace objects and recreate solution */
    if (adapted) {
      const char *name;

      CHKERRQ(PetscObjectGetName((PetscObject) dm, &name));
      CHKERRQ(PetscObjectSetName((PetscObject) odm, name));
      /* Reconfigure solver */
      CHKERRQ(SNESReset(adaptor->snes));
      CHKERRQ(SNESSetDM(adaptor->snes, odm));
      CHKERRQ(DMAdaptorSetSolver(adaptor, adaptor->snes));
      CHKERRQ(DMPlexSetSNESLocalFEM(odm, ctx, ctx, ctx));
      CHKERRQ(SNESSetFromOptions(adaptor->snes));
      /* Transfer system */
      CHKERRQ(DMCopyDisc(dm, odm));
      /* Transfer solution */
      CHKERRQ(DMCreateGlobalVector(odm, &ox));
      CHKERRQ(PetscObjectGetName((PetscObject) x, &name));
      CHKERRQ(PetscObjectSetName((PetscObject) ox, name));
      CHKERRQ(DMAdaptorTransferSolution(adaptor, dm, x, odm, ox));
      /* Cleanup adaptivity info */
      if (adaptIter > 0) CHKERRQ(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_(comm)));
      CHKERRQ(DMForestSetAdaptivityForest(dm, NULL)); /* clear internal references to the previous dm */
      CHKERRQ(DMDestroy(&dm));
      CHKERRQ(VecDestroy(&x));
      *adm = odm;
      *ax  = ox;
    } else {
      *adm = dm;
      *ax  = x;
      adaptIter = numAdapt;
    }
    if (adaptIter < numAdapt-1) {
      CHKERRQ(DMViewFromOptions(odm, NULL, "-dm_adapt_iter_view"));
      CHKERRQ(VecViewFromOptions(ox, NULL, "-sol_adapt_iter_view"));
    }
  }
  CHKERRQ(DMViewFromOptions(*adm, NULL, "-dm_adapt_view"));
  CHKERRQ(VecViewFromOptions(*ax, NULL, "-sol_adapt_view"));
  PetscFunctionReturn(0);
}

/*@
  DMAdaptorAdapt - Creates a new DM that is adapted to the problem

  Not collective

  Input Parameters:
+ adaptor  - The DMAdaptor object
. x        - The global approximate solution
- strategy - The adaptation strategy

  Output Parameters:
+ adm - The adapted DM
- ax  - The adapted solution

  Options database keys:
+ -snes_adapt <strategy> - initial, sequential, multigrid
. -adapt_gradient_view - View the Clement interpolant of the solution gradient
. -adapt_hessian_view - View the Clement interpolant of the solution Hessian
- -adapt_metric_view - View the metric tensor for adaptive mesh refinement

  Note: The available adaptation strategies are:
$ 1) Adapt the initial mesh until a quality metric, e.g., a priori error bound, is satisfied
$ 2) Solve the problem on a series of adapted meshes until a quality metric, e.g. a posteriori error bound, is satisfied
$ 3) Solve the problem on a hierarchy of adapted meshes generated to satisfy a quality metric using multigrid

  Level: intermediate

.seealso: DMAdaptorSetSolver(), DMAdaptorCreate(), DMAdaptorAdapt()
@*/
PetscErrorCode DMAdaptorAdapt(DMAdaptor adaptor, Vec x, DMAdaptationStrategy strategy, DM *adm, Vec *ax)
{
  PetscFunctionBegin;
  switch (strategy)
  {
  case DM_ADAPTATION_INITIAL:
    CHKERRQ(DMAdaptorAdapt_Sequence_Private(adaptor, x, PETSC_FALSE, adm, ax));
    break;
  case DM_ADAPTATION_SEQUENTIAL:
    CHKERRQ(DMAdaptorAdapt_Sequence_Private(adaptor, x, PETSC_TRUE, adm, ax));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject) adaptor), PETSC_ERR_ARG_WRONG, "Unrecognized adaptation strategy %d", strategy);
  }
  PetscFunctionReturn(0);
}
