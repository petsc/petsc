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
  PetscCall(DMProjectFunction(adm, 0.0, adaptor->exactSol, adaptor->exactCtx, INSERT_ALL_VALUES, au));
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
  PetscCall(PetscSysInitializePackage());
  PetscCall(PetscHeaderCreate(*adaptor, DM_CLASSID, "DMAdaptor", "DM Adaptor", "SNES", comm, DMAdaptorDestroy, DMAdaptorView));

  (*adaptor)->monitor = PETSC_FALSE;
  (*adaptor)->adaptCriterion = DM_ADAPTATION_NONE;
  (*adaptor)->numSeq = 1;
  (*adaptor)->Nadapt = -1;
  (*adaptor)->refinementFactor = 2.0;
  (*adaptor)->ops->computeerrorindicator = DMAdaptorSimpleErrorIndicator_Private;
  refineBox.min = refineBox.max = PETSC_MAX_REAL;
  PetscCall(VecTaggerCreate(PetscObjectComm((PetscObject) *adaptor), &(*adaptor)->refineTag));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) (*adaptor)->refineTag, "refine_"));
  PetscCall(VecTaggerSetType((*adaptor)->refineTag, VECTAGGERABSOLUTE));
  PetscCall(VecTaggerAbsoluteSetBox((*adaptor)->refineTag, &refineBox));
  coarsenBox.min = coarsenBox.max = PETSC_MAX_REAL;
  PetscCall(VecTaggerCreate(PetscObjectComm((PetscObject) *adaptor), &(*adaptor)->coarsenTag));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) (*adaptor)->coarsenTag, "coarsen_"));
  PetscCall(VecTaggerSetType((*adaptor)->coarsenTag, VECTAGGERABSOLUTE));
  PetscCall(VecTaggerAbsoluteSetBox((*adaptor)->coarsenTag, &coarsenBox));
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
  PetscCall(VecTaggerDestroy(&(*adaptor)->refineTag));
  PetscCall(VecTaggerDestroy(&(*adaptor)->coarsenTag));
  PetscCall(PetscFree2((*adaptor)->exactSol, (*adaptor)->exactCtx));
  PetscCall(PetscHeaderDestroy(adaptor));
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
  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject) adaptor), "", "DM Adaptor Options", "DMAdaptor");
  PetscCall(PetscOptionsBool("-adaptor_monitor", "Monitor the adaptation process", "DMAdaptorMonitor", adaptor->monitor, &adaptor->monitor, NULL));
  PetscCall(PetscOptionsInt("-adaptor_sequence_num", "Number of adaptations to generate an optimal grid", "DMAdaptorSetSequenceLength", adaptor->numSeq, &adaptor->numSeq, NULL));
  PetscCall(PetscOptionsInt("-adaptor_target_num", "Set the target number of vertices N_adapt, -1 for automatic determination", "DMAdaptor", adaptor->Nadapt, &adaptor->Nadapt, NULL));
  PetscCall(PetscOptionsReal("-adaptor_refinement_factor", "Set r such that N_adapt = r^dim N_orig", "DMAdaptor", adaptor->refinementFactor, &adaptor->refinementFactor, NULL));
  PetscOptionsEnd();
  PetscCall(VecTaggerSetFromOptions(adaptor->refineTag));
  PetscCall(VecTaggerSetFromOptions(adaptor->coarsenTag));
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
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject) adaptor, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "DM Adaptor\n"));
  PetscCall(PetscViewerASCIIPrintf(viewer, "  sequence length: %" PetscInt_FMT "\n", adaptor->numSeq));
  PetscCall(VecTaggerView(adaptor->refineTag,  viewer));
  PetscCall(VecTaggerView(adaptor->coarsenTag, viewer));
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
  PetscCall(SNESGetDM(adaptor->snes, &adaptor->idm));
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
  PetscValidIntPointer(num, 2);
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
  PetscCall(DMGetDS(adaptor->idm, &prob));
  PetscCall(VecTaggerSetUp(adaptor->refineTag));
  PetscCall(VecTaggerSetUp(adaptor->coarsenTag));
  PetscCall(PetscDSGetNumFields(prob, &Nf));
  PetscCall(PetscMalloc2(Nf, &adaptor->exactSol, Nf, &adaptor->exactCtx));
  for (f = 0; f < Nf; ++f) {
    PetscCall(PetscDSGetExactSolution(prob, f, &adaptor->exactSol[f], &adaptor->exactCtx[f]));
    /* TODO Have a flag that forces projection rather than using the exact solution */
    if (adaptor->exactSol[0]) PetscCall(DMAdaptorSetTransferFunction(adaptor, DMAdaptorTransferSolution_Exact_Private));
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
  PetscCall(DMConvert(adaptor->idm, DMPLEX, &plex));
  PetscCall(DMGetDS(adaptor->idm, &prob));
  PetscCall(PetscDSGetDiscretization(prob, 0, &obj));
  PetscCall(PetscObjectGetClassId(obj, &id));
  PetscCall(DMIsForest(adaptor->idm, &isForest));
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
    PetscCall(DMPlexInsertBoundaryValues(plex, PETSC_TRUE, locX, 0.0, adaptor->faceGeom, adaptor->cellGeom, NULL));
  } else {
    PetscFV      fvm = (PetscFV) obj;
    PetscLimiter noneLimiter;
    Vec          grad;

    PetscCall(PetscFVGetComputeGradients(fvm, &adaptor->computeGradient));
    PetscCall(PetscFVSetComputeGradients(fvm, PETSC_TRUE));
    /* Use no limiting when reconstructing gradients for adaptivity */
    PetscCall(PetscFVGetLimiter(fvm, &adaptor->limiter));
    PetscCall(PetscObjectReference((PetscObject) adaptor->limiter));
    PetscCall(PetscLimiterCreate(PetscObjectComm((PetscObject) fvm), &noneLimiter));
    PetscCall(PetscLimiterSetType(noneLimiter, PETSCLIMITERNONE));
    PetscCall(PetscFVSetLimiter(fvm, noneLimiter));
    /* Get FVM data */
    PetscCall(DMPlexGetDataFVM(plex, fvm, &adaptor->cellGeom, &adaptor->faceGeom, &adaptor->gradDM));
    PetscCall(VecGetDM(adaptor->cellGeom, &adaptor->cellDM));
    PetscCall(VecGetArrayRead(adaptor->cellGeom, &adaptor->cellGeomArray));
    /* Compute local solution bc */
    PetscCall(DMPlexInsertBoundaryValues(plex, PETSC_TRUE, locX, 0.0, adaptor->faceGeom, adaptor->cellGeom, NULL));
    /* Compute gradients */
    PetscCall(DMCreateGlobalVector(adaptor->gradDM, &grad));
    PetscCall(DMPlexReconstructGradientsFVM(plex, locX, grad));
    PetscCall(DMGetLocalVector(adaptor->gradDM, &adaptor->cellGrad));
    PetscCall(DMGlobalToLocalBegin(adaptor->gradDM, grad, INSERT_VALUES, adaptor->cellGrad));
    PetscCall(DMGlobalToLocalEnd(adaptor->gradDM, grad, INSERT_VALUES, adaptor->cellGrad));
    PetscCall(VecDestroy(&grad));
    PetscCall(VecGetArrayRead(adaptor->cellGrad, &adaptor->cellGradArray));
  }
  PetscCall(DMDestroy(&plex));
  PetscFunctionReturn(0);
}

PetscErrorCode DMAdaptorTransferSolution(DMAdaptor adaptor, DM dm, Vec x, DM adm, Vec ax)
{
  PetscReal      time = 0.0;
  Mat            interp;
  void          *ctx;

  PetscFunctionBegin;
  PetscCall(DMGetApplicationContext(dm, &ctx));
  if (adaptor->ops->transfersolution) {
    PetscCall((*adaptor->ops->transfersolution)(adaptor, dm, x, adm, ax, ctx));
  } else {
    switch (adaptor->adaptCriterion) {
    case DM_ADAPTATION_LABEL:
      PetscCall(DMForestTransferVec(dm, x, adm, ax, PETSC_TRUE, time));
      break;
    case DM_ADAPTATION_REFINE:
    case DM_ADAPTATION_METRIC:
      PetscCall(DMCreateInterpolation(dm, adm, &interp, NULL));
      PetscCall(MatInterpolate(interp, x, ax));
      PetscCall(DMInterpolate(dm, interp, adm));
      PetscCall(MatDestroy(&interp));
      break;
    default: SETERRQ(PetscObjectComm((PetscObject) adaptor), PETSC_ERR_SUP, "No built-in projection for this adaptation criterion: %d", adaptor->adaptCriterion);
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
  PetscCall(DMGetDS(adaptor->idm, &prob));
  PetscCall(PetscDSGetDiscretization(prob, 0, &obj));
  PetscCall(PetscObjectGetClassId(obj, &id));
  if (id == PETSCFV_CLASSID) {
    PetscFV fvm = (PetscFV) obj;

    PetscCall(PetscFVSetComputeGradients(fvm, adaptor->computeGradient));
    /* Restore original limiter */
    PetscCall(PetscFVSetLimiter(fvm, adaptor->limiter));

    PetscCall(VecRestoreArrayRead(adaptor->cellGeom, &adaptor->cellGeomArray));
    PetscCall(VecRestoreArrayRead(adaptor->cellGrad, &adaptor->cellGradArray));
    PetscCall(DMRestoreLocalVector(adaptor->gradDM, &adaptor->cellGrad));
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
  PetscCall(DMGetDimension(plex, &dim));
  PetscCall(DMGetCoordinateDim(plex, &cdim));
  PetscCall(DMGetApplicationContext(plex, &ctx));
  PetscCall(DMGetDS(plex, &prob));
  PetscCall(PetscDSGetDiscretization(prob, 0, &obj));
  PetscCall(PetscObjectGetClassId(obj, &id));
  if (id == PETSCFV_CLASSID) {
    const PetscScalar *pointSols;
    const PetscScalar *pointSol;
    const PetscScalar *pointGrad;
    PetscFVCellGeom   *cg;

    PetscCall(PetscFVGetNumComponents((PetscFV) obj, &Nc));
    PetscCall(VecGetArrayRead(locX, &pointSols));
    PetscCall(DMPlexPointLocalRead(plex, cell, pointSols, (void *) &pointSol));
    PetscCall(DMPlexPointLocalRead(adaptor->gradDM, cell, adaptor->cellGradArray, (void *) &pointGrad));
    PetscCall(DMPlexPointLocalRead(adaptor->cellDM, cell, adaptor->cellGeomArray, &cg));
    PetscCall((*adaptor->ops->computeerrorindicator)(adaptor, dim, Nc, pointSol, pointGrad, cg, errInd, ctx));
    PetscCall(VecRestoreArrayRead(locX, &pointSols));
  } else {
    PetscScalar     *x = NULL, *field, *gradient, *interpolant, *interpolantGrad;
    PetscFVCellGeom  cg;
    PetscFEGeom      fegeom;
    const PetscReal *quadWeights;
    PetscReal       *coords;
    PetscInt         Nb, fc, Nq, qNc, Nf, f, fieldOffset;

    fegeom.dim      = dim;
    fegeom.dimEmbed = cdim;
    PetscCall(PetscDSGetNumFields(prob, &Nf));
    PetscCall(PetscFEGetQuadrature((PetscFE) obj, &quad));
    PetscCall(DMPlexVecGetClosure(plex, NULL, locX, cell, NULL, &x));
    PetscCall(PetscFEGetDimension((PetscFE) obj, &Nb));
    PetscCall(PetscFEGetNumComponents((PetscFE) obj, &Nc));
    PetscCall(PetscQuadratureGetData(quad, NULL, &qNc, &Nq, NULL, &quadWeights));
    PetscCall(PetscMalloc6(Nc,&field,cdim*Nc,&gradient,cdim*Nq,&coords,Nq,&fegeom.detJ,cdim*cdim*Nq,&fegeom.J,cdim*cdim*Nq,&fegeom.invJ));
    PetscCall(PetscMalloc2(Nc, &interpolant, cdim*Nc, &interpolantGrad));
    PetscCall(DMPlexComputeCellGeometryFEM(plex, cell, quad, coords, fegeom.J, fegeom.invJ, fegeom.detJ));
    PetscCall(DMPlexComputeCellGeometryFVM(plex, cell, &cg.volume, NULL, NULL));
    PetscCall(PetscArrayzero(gradient, cdim*Nc));
    for (f = 0, fieldOffset = 0; f < Nf; ++f) {
      PetscInt qc = 0, q;

      PetscCall(PetscDSGetDiscretization(prob, f, &obj));
      PetscCall(PetscArrayzero(interpolant,Nc));
      PetscCall(PetscArrayzero(interpolantGrad, cdim*Nc));
      for (q = 0; q < Nq; ++q) {
        PetscCall(PetscFEInterpolateFieldAndGradient_Static((PetscFE) obj, 1, x, &fegeom, q, interpolant, interpolantGrad));
        for (fc = 0; fc < Nc; ++fc) {
          const PetscReal wt = quadWeights[q*qNc+qc+fc];

          field[fc] += interpolant[fc]*wt*fegeom.detJ[q];
          for (d = 0; d < cdim; ++d) gradient[fc*cdim+d] += interpolantGrad[fc*dim+d]*wt*fegeom.detJ[q];
        }
      }
      fieldOffset += Nb;
      qc          += Nc;
    }
    PetscCall(PetscFree2(interpolant, interpolantGrad));
    PetscCall(DMPlexVecRestoreClosure(plex, NULL, locX, cell, NULL, &x));
    for (fc = 0; fc < Nc; ++fc) {
      field[fc] /= cg.volume;
      for (d = 0; d < cdim; ++d) gradient[fc*cdim+d] /= cg.volume;
    }
    PetscCall((*adaptor->ops->computeerrorindicator)(adaptor, dim, Nc, field, gradient, &cg, errInd, ctx));
    PetscCall(PetscFree6(field,gradient,coords,fegeom.detJ,fegeom.J,fegeom.invJ));
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
  PetscCall(DMViewFromOptions(adaptor->idm, NULL, "-dm_adapt_pre_view"));
  PetscCall(VecViewFromOptions(inx, NULL, "-sol_adapt_pre_view"));
  PetscCall(PetscObjectGetComm((PetscObject) adaptor, &comm));
  PetscCall(DMGetDimension(adaptor->idm, &dim));
  PetscCall(DMGetCoordinateDim(adaptor->idm, &coordDim));
  PetscCall(DMGetApplicationContext(adaptor->idm, &ctx));
  PetscCall(DMGetDS(adaptor->idm, &prob));
  PetscCall(PetscDSGetNumFields(prob, &numFields));
  PetscCheck(numFields != 0,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Number of fields is zero!");

  /* Adapt until nothing changes */
  /* Adapt for a specified number of iterates */
  for (adaptIter = 0; adaptIter < numAdapt-1; ++adaptIter) PetscCall(PetscViewerASCIIPushTab(PETSC_VIEWER_STDOUT_(comm)));
  for (adaptIter = 0; adaptIter < numAdapt;   ++adaptIter) {
    PetscBool adapted = PETSC_FALSE;
    DM        dm      = adaptIter ? *adm : adaptor->idm, odm;
    Vec       x       = adaptIter ? *ax  : inx, locX, ox;

    PetscCall(DMGetLocalVector(dm, &locX));
    PetscCall(DMGlobalToLocalBegin(dm, adaptIter ? *ax : x, INSERT_VALUES, locX));
    PetscCall(DMGlobalToLocalEnd(dm, adaptIter ? *ax : x, INSERT_VALUES, locX));
    PetscCall(DMAdaptorPreAdapt(adaptor, locX));
    if (doSolve) {
      SNES snes;

      PetscCall(DMAdaptorGetSolver(adaptor, &snes));
      PetscCall(SNESSolve(snes, NULL, adaptIter ? *ax : x));
    }
    /* PetscCall(DMAdaptorMonitor(adaptor));
       Print iterate, memory used, DM, solution */
    switch (adaptor->adaptCriterion) {
    case DM_ADAPTATION_REFINE:
      PetscCall(DMRefine(dm, comm, &odm));
      PetscCheck(odm,comm, PETSC_ERR_ARG_INCOMP, "DMRefine() did not perform any refinement, cannot continue grid sequencing");
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

      PetscCall(DMConvert(dm, DMPLEX, &plex));
      PetscCall(DMLabelCreate(PETSC_COMM_SELF, "adapt", &adaptLabel));
      PetscCall(DMPlexGetSimplexOrBoxCells(plex, 0, &cStart, &cEnd));

      PetscCall(VecCreateMPI(PetscObjectComm((PetscObject) adaptor), cEnd-cStart, PETSC_DETERMINE, &errVec));
      PetscCall(VecSetUp(errVec));
      PetscCall(VecGetArray(errVec, &errArray));
      PetscCall(VecGetArrayRead(locX, &pointSols));
      for (c = cStart; c < cEnd; ++c) {
        PetscReal errInd;

        PetscCall(DMAdaptorComputeErrorIndicator_Private(adaptor, plex, c, locX, &errInd));
        errArray[c-cStart] = errInd;
        minMaxInd[0] = PetscMin(minMaxInd[0], errInd);
        minMaxInd[1] = PetscMax(minMaxInd[1], errInd);
      }
      PetscCall(VecRestoreArrayRead(locX, &pointSols));
      PetscCall(VecRestoreArray(errVec, &errArray));
      PetscCall(PetscGlobalMinMaxReal(PetscObjectComm((PetscObject) adaptor), minMaxInd, minMaxIndGlobal));
      PetscCall(PetscInfo(adaptor, "DMAdaptor: error indicator range (%g, %g)\n", (double)minMaxIndGlobal[0], (double)minMaxIndGlobal[1]));
      /*     Compute IS from VecTagger */
      PetscCall(VecTaggerComputeIS(adaptor->refineTag, errVec, &refineIS,NULL));
      PetscCall(VecTaggerComputeIS(adaptor->coarsenTag, errVec, &coarsenIS,NULL));
      PetscCall(ISGetSize(refineIS, &nRefine));
      PetscCall(ISGetSize(coarsenIS, &nCoarsen));
      PetscCall(PetscInfo(adaptor, "DMAdaptor: numRefine %" PetscInt_FMT ", numCoarsen %" PetscInt_FMT "\n", nRefine, nCoarsen));
      if (nRefine)  PetscCall(DMLabelSetStratumIS(adaptLabel, DM_ADAPT_REFINE,  refineIS));
      if (nCoarsen) PetscCall(DMLabelSetStratumIS(adaptLabel, DM_ADAPT_COARSEN, coarsenIS));
      PetscCall(ISDestroy(&coarsenIS));
      PetscCall(ISDestroy(&refineIS));
      PetscCall(VecDestroy(&errVec));
      /*     Adapt DM from label */
      if (nRefine || nCoarsen) {
        PetscCall(DMAdaptLabel(dm, adaptLabel, &odm));
        adapted = PETSC_TRUE;
      }
      PetscCall(DMLabelDestroy(&adaptLabel));
      PetscCall(DMDestroy(&plex));
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

      PetscCall(PetscMalloc(1, &funcs));
      funcs[0] = identityFunc;

      /*     Setup finite element spaces */
      PetscCall(DMClone(dm, &dmGrad));
      PetscCall(DMClone(dm, &dmHess));
      PetscCheck(numFields <= 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Adaptation with multiple fields not yet considered");  // TODO
      for (f = 0; f < numFields; ++f) {
        PetscFE         fe, feGrad, feHess;
        PetscDualSpace  Q;
        PetscSpace      space;
        DM              K;
        PetscQuadrature q;
        PetscInt        Nc, qorder, p;
        const char     *prefix;

        PetscCall(PetscDSGetDiscretization(prob, f, (PetscObject *) &fe));
        PetscCall(PetscFEGetNumComponents(fe, &Nc));
        PetscCheck(Nc <= 1,PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Adaptation with multiple components not yet considered");  // TODO
        PetscCall(PetscFEGetBasisSpace(fe, &space));
        PetscCall(PetscSpaceGetDegree(space, NULL, &p));
        if (p > 1) higherOrder = PETSC_TRUE;
        PetscCall(PetscFEGetDualSpace(fe, &Q));
        PetscCall(PetscDualSpaceGetDM(Q, &K));
        PetscCall(DMPlexGetDepthStratum(K, 0, &vStart, &vEnd));
        PetscCall(PetscFEGetQuadrature(fe, &q));
        PetscCall(PetscQuadratureGetOrder(q, &qorder));
        PetscCall(PetscObjectGetOptionsPrefix((PetscObject) fe, &prefix));
        PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject) dmGrad), dim, Nc*coordDim, PETSC_TRUE, prefix, qorder, &feGrad));
        PetscCall(PetscFECreateDefault(PetscObjectComm((PetscObject) dmHess), dim, Nc*Nd, PETSC_TRUE, prefix, qorder, &feHess));
        PetscCall(DMSetField(dmGrad, f, NULL, (PetscObject)feGrad));
        PetscCall(DMSetField(dmHess, f, NULL, (PetscObject)feHess));
        PetscCall(DMCreateDS(dmGrad));
        PetscCall(DMCreateDS(dmHess));
        PetscCall(PetscFEDestroy(&feGrad));
        PetscCall(PetscFEDestroy(&feHess));
      }
      /*     Compute vertexwise gradients from cellwise gradients */
      PetscCall(DMCreateLocalVector(dmGrad, &xGrad));
      PetscCall(VecViewFromOptions(locX, NULL, "-sol_adapt_loc_pre_view"));
      PetscCall(DMPlexComputeGradientClementInterpolant(dm, locX, xGrad));
      PetscCall(VecViewFromOptions(xGrad, NULL, "-adapt_gradient_view"));
      /*     Compute vertexwise Hessians from cellwise Hessians */
      PetscCall(DMCreateLocalVector(dmHess, &xHess));
      PetscCall(DMPlexComputeGradientClementInterpolant(dmGrad, xGrad, xHess));
      PetscCall(VecViewFromOptions(xHess, NULL, "-adapt_hessian_view"));
      PetscCall(VecDestroy(&xGrad));
      PetscCall(DMDestroy(&dmGrad));
      /*     Compute L-p normalized metric */
      PetscCall(DMClone(dm, &dmMetric));
      N    = adaptor->Nadapt >= 0 ? adaptor->Nadapt : PetscPowRealInt(adaptor->refinementFactor, dim)*((PetscReal) (vEnd - vStart));
      if (adaptor->monitor) {
        PetscMPIInt rank, size;
        PetscCallMPI(MPI_Comm_rank(comm, &size));
        PetscCallMPI(MPI_Comm_rank(comm, &rank));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "[%d] N_orig: %" PetscInt_FMT " N_adapt: %g\n", rank, vEnd - vStart, (double)N));
      }
      PetscCall(DMPlexMetricSetTargetComplexity(dmMetric, (PetscReal) N));
      if (higherOrder) {
        /*   Project Hessian into P1 space, if required */
        PetscCall(DMPlexMetricCreate(dmMetric, 0, &metric));
        PetscCall(DMProjectFieldLocal(dmMetric, 0.0, xHess, funcs, INSERT_ALL_VALUES, metric));
        PetscCall(VecDestroy(&xHess));
        xHess = metric;
      }
      PetscCall(PetscFree(funcs));
      PetscCall(DMPlexMetricNormalize(dmMetric, xHess, PETSC_TRUE, PETSC_TRUE, &metric));
      PetscCall(VecDestroy(&xHess));
      PetscCall(DMDestroy(&dmHess));
      /*     Adapt DM from metric */
      PetscCall(DMGetLabel(dm, "marker", &bdLabel));
      PetscCall(DMAdaptMetric(dm, metric, bdLabel, rgLabel, &odm));
      adapted = PETSC_TRUE;
      /*     Cleanup */
      PetscCall(VecDestroy(&metric));
      PetscCall(DMDestroy(&dmMetric));
    }
    break;
    default: SETERRQ(comm, PETSC_ERR_ARG_WRONG, "Invalid adaptation type: %d", adaptor->adaptCriterion);
    }
    PetscCall(DMAdaptorPostAdapt(adaptor));
    PetscCall(DMRestoreLocalVector(dm, &locX));
    /* If DM was adapted, replace objects and recreate solution */
    if (adapted) {
      const char *name;

      PetscCall(PetscObjectGetName((PetscObject) dm, &name));
      PetscCall(PetscObjectSetName((PetscObject) odm, name));
      /* Reconfigure solver */
      PetscCall(SNESReset(adaptor->snes));
      PetscCall(SNESSetDM(adaptor->snes, odm));
      PetscCall(DMAdaptorSetSolver(adaptor, adaptor->snes));
      PetscCall(DMPlexSetSNESLocalFEM(odm, ctx, ctx, ctx));
      PetscCall(SNESSetFromOptions(adaptor->snes));
      /* Transfer system */
      PetscCall(DMCopyDisc(dm, odm));
      /* Transfer solution */
      PetscCall(DMCreateGlobalVector(odm, &ox));
      PetscCall(PetscObjectGetName((PetscObject) x, &name));
      PetscCall(PetscObjectSetName((PetscObject) ox, name));
      PetscCall(DMAdaptorTransferSolution(adaptor, dm, x, odm, ox));
      /* Cleanup adaptivity info */
      if (adaptIter > 0) PetscCall(PetscViewerASCIIPopTab(PETSC_VIEWER_STDOUT_(comm)));
      PetscCall(DMForestSetAdaptivityForest(dm, NULL)); /* clear internal references to the previous dm */
      PetscCall(DMDestroy(&dm));
      PetscCall(VecDestroy(&x));
      *adm = odm;
      *ax  = ox;
    } else {
      *adm = dm;
      *ax  = x;
      adaptIter = numAdapt;
    }
    if (adaptIter < numAdapt-1) {
      PetscCall(DMViewFromOptions(odm, NULL, "-dm_adapt_iter_view"));
      PetscCall(VecViewFromOptions(ox, NULL, "-sol_adapt_iter_view"));
    }
  }
  PetscCall(DMViewFromOptions(*adm, NULL, "-dm_adapt_view"));
  PetscCall(VecViewFromOptions(*ax, NULL, "-sol_adapt_view"));
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
    PetscCall(DMAdaptorAdapt_Sequence_Private(adaptor, x, PETSC_FALSE, adm, ax));
    break;
  case DM_ADAPTATION_SEQUENTIAL:
    PetscCall(DMAdaptorAdapt_Sequence_Private(adaptor, x, PETSC_TRUE, adm, ax));
    break;
  default: SETERRQ(PetscObjectComm((PetscObject) adaptor), PETSC_ERR_ARG_WRONG, "Unrecognized adaptation strategy %d", strategy);
  }
  PetscFunctionReturn(0);
}
