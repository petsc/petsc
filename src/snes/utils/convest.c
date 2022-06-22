#include <petscconvest.h>            /*I "petscconvest.h" I*/
#include <petscdmplex.h>
#include <petscds.h>

#include <petsc/private/petscconvestimpl.h>

static PetscErrorCode zero_private(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = 0.0;
  return 0;
}

/*@
  PetscConvEstDestroy - Destroys a PetscConvEst object

  Collective on PetscConvEst

  Input Parameter:
. ce - The PetscConvEst object

  Level: beginner

.seealso: `PetscConvEstCreate()`, `PetscConvEstGetConvRate()`
@*/
PetscErrorCode PetscConvEstDestroy(PetscConvEst *ce)
{
  PetscFunctionBegin;
  if (!*ce) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*ce),PETSC_OBJECT_CLASSID,1);
  if (--((PetscObject)(*ce))->refct > 0) {
    *ce = NULL;
    PetscFunctionReturn(0);
  }
  PetscCall(PetscFree3((*ce)->initGuess, (*ce)->exactSol, (*ce)->ctxs));
  PetscCall(PetscFree2((*ce)->dofs, (*ce)->errors));
  PetscCall(PetscHeaderDestroy(ce));
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstSetFromOptions - Sets a PetscConvEst object from options

  Collective on PetscConvEst

  Input Parameters:
. ce - The PetscConvEst object

  Level: beginner

.seealso: `PetscConvEstCreate()`, `PetscConvEstGetConvRate()`
@*/
PetscErrorCode PetscConvEstSetFromOptions(PetscConvEst ce)
{
  PetscFunctionBegin;
  PetscOptionsBegin(PetscObjectComm((PetscObject) ce), "", "Convergence Estimator Options", "PetscConvEst");
  PetscCall(PetscOptionsInt("-convest_num_refine", "The number of refinements for the convergence check", "PetscConvEst", ce->Nr, &ce->Nr, NULL));
  PetscCall(PetscOptionsReal("-convest_refine_factor", "The increase in resolution in each dimension", "PetscConvEst", ce->r, &ce->r, NULL));
  PetscCall(PetscOptionsBool("-convest_monitor", "Monitor the error for each convergence check", "PetscConvEst", ce->monitor, &ce->monitor, NULL));
  PetscCall(PetscOptionsBool("-convest_no_refine", "Debugging flag to run on the same mesh each time", "PetscConvEst", ce->noRefine, &ce->noRefine, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstView - Views a PetscConvEst object

  Collective on PetscConvEst

  Input Parameters:
+ ce     - The PetscConvEst object
- viewer - The PetscViewer object

  Level: beginner

.seealso: `PetscConvEstCreate()`, `PetscConvEstGetConvRate()`
@*/
PetscErrorCode PetscConvEstView(PetscConvEst ce, PetscViewer viewer)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectPrintClassNamePrefixType((PetscObject) ce, viewer));
  PetscCall(PetscViewerASCIIPrintf(viewer, "ConvEst with %" PetscInt_FMT " levels\n", ce->Nr+1));
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstGetSolver - Gets the solver used to produce discrete solutions

  Not collective

  Input Parameter:
. ce     - The PetscConvEst object

  Output Parameter:
. solver - The solver

  Level: intermediate

.seealso: `PetscConvEstSetSolver()`, `PetscConvEstCreate()`, `PetscConvEstGetConvRate()`
@*/
PetscErrorCode PetscConvEstGetSolver(PetscConvEst ce, PetscObject *solver)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ce, PETSC_OBJECT_CLASSID, 1);
  PetscValidPointer(solver, 2);
  *solver = ce->solver;
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstSetSolver - Sets the solver used to produce discrete solutions

  Not collective

  Input Parameters:
+ ce     - The PetscConvEst object
- solver - The solver

  Level: intermediate

  Note: The solver MUST have an attached DM/DS, so that we know the exact solution

.seealso: `PetscConvEstGetSNES()`, `PetscConvEstCreate()`, `PetscConvEstGetConvRate()`
@*/
PetscErrorCode PetscConvEstSetSolver(PetscConvEst ce, PetscObject solver)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ce, PETSC_OBJECT_CLASSID, 1);
  PetscValidHeader(solver, 2);
  ce->solver = solver;
  PetscCall((*ce->ops->setsolver)(ce, solver));
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstSetUp - After the solver is specified, we create structures for estimating convergence

  Collective on PetscConvEst

  Input Parameters:
. ce - The PetscConvEst object

  Level: beginner

.seealso: `PetscConvEstCreate()`, `PetscConvEstGetConvRate()`
@*/
PetscErrorCode PetscConvEstSetUp(PetscConvEst ce)
{
  PetscInt       Nf, f, Nds, s;

  PetscFunctionBegin;
  PetscCall(DMGetNumFields(ce->idm, &Nf));
  ce->Nf = PetscMax(Nf, 1);
  PetscCall(PetscMalloc2((ce->Nr+1)*ce->Nf, &ce->dofs, (ce->Nr+1)*ce->Nf, &ce->errors));
  PetscCall(PetscCalloc3(ce->Nf, &ce->initGuess, ce->Nf, &ce->exactSol, ce->Nf, &ce->ctxs));
  for (f = 0; f < Nf; ++f) ce->initGuess[f] = zero_private;
  PetscCall(DMGetNumDS(ce->idm, &Nds));
  for (s = 0; s < Nds; ++s) {
    PetscDS         ds;
    DMLabel         label;
    IS              fieldIS;
    const PetscInt *fields;
    PetscInt        dsNf;

    PetscCall(DMGetRegionNumDS(ce->idm, s, &label, &fieldIS, &ds));
    PetscCall(PetscDSGetNumFields(ds, &dsNf));
    if (fieldIS) PetscCall(ISGetIndices(fieldIS, &fields));
    for (f = 0; f < dsNf; ++f) {
      const PetscInt field = fields[f];
      PetscCall(PetscDSGetExactSolution(ds, field, &ce->exactSol[field], &ce->ctxs[field]));
    }
    if (fieldIS) PetscCall(ISRestoreIndices(fieldIS, &fields));
  }
  for (f = 0; f < Nf; ++f) {
    PetscCheck(ce->exactSol[f],PetscObjectComm((PetscObject) ce), PETSC_ERR_ARG_WRONG, "DS must contain exact solution functions in order to estimate convergence, missing for field %" PetscInt_FMT, f);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode PetscConvEstComputeInitialGuess(PetscConvEst ce, PetscInt r, DM dm, Vec u)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ce, PETSC_OBJECT_CLASSID, 1);
  if (dm) PetscValidHeaderSpecific(dm, DM_CLASSID, 3);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 4);
  PetscCall((*ce->ops->initguess)(ce, r, dm, u));
  PetscFunctionReturn(0);
}

PetscErrorCode PetscConvEstComputeError(PetscConvEst ce, PetscInt r, DM dm, Vec u, PetscReal errors[])
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ce, PETSC_OBJECT_CLASSID, 1);
  if (dm) PetscValidHeaderSpecific(dm, DM_CLASSID, 3);
  PetscValidHeaderSpecific(u, VEC_CLASSID, 4);
  PetscValidRealPointer(errors, 5);
  PetscCall((*ce->ops->computeerror)(ce, r, dm, u, errors));
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstMonitorDefault - Monitors the convergence estimation loop

  Collective on PetscConvEst

  Input Parameters:
+ ce - The PetscConvEst object
- r  - The refinement level

  Options database keys:
. -convest_monitor - Activate the monitor

  Level: intermediate

.seealso: `PetscConvEstCreate()`, `PetscConvEstGetConvRate()`, `SNESSolve()`, `TSSolve()`
@*/
PetscErrorCode PetscConvEstMonitorDefault(PetscConvEst ce, PetscInt r)
{
  MPI_Comm       comm;
  PetscInt       f;

  PetscFunctionBegin;
  if (ce->monitor) {
    PetscInt  *dofs   = &ce->dofs[r*ce->Nf];
    PetscReal *errors = &ce->errors[r*ce->Nf];

    PetscCall(PetscObjectGetComm((PetscObject) ce, &comm));
    PetscCall(PetscPrintf(comm, "N: "));
    if (ce->Nf > 1) PetscCall(PetscPrintf(comm, "["));
    for (f = 0; f < ce->Nf; ++f) {
      if (f > 0) PetscCall(PetscPrintf(comm, ", "));
      PetscCall(PetscPrintf(comm, "%7" PetscInt_FMT, dofs[f]));
    }
    if (ce->Nf > 1) PetscCall(PetscPrintf(comm, "]"));
    PetscCall(PetscPrintf(comm, "  "));
    PetscCall(PetscPrintf(comm, "L_2 Error: "));
    if (ce->Nf > 1) PetscCall(PetscPrintf(comm, "["));
    for (f = 0; f < ce->Nf; ++f) {
      if (f > 0) PetscCall(PetscPrintf(comm, ", "));
      if (errors[f] < 1.0e-11) PetscCall(PetscPrintf(comm, "< 1e-11"));
      else                     PetscCall(PetscPrintf(comm, "%g", (double) errors[f]));
    }
    if (ce->Nf > 1) PetscCall(PetscPrintf(comm, "]"));
    PetscCall(PetscPrintf(comm, "\n"));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstSetSNES_Private(PetscConvEst ce, PetscObject solver)
{
  PetscClassId   id;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetClassId(ce->solver, &id));
  PetscCheck(id == SNES_CLASSID,PetscObjectComm((PetscObject) ce), PETSC_ERR_ARG_WRONG, "Solver was not a SNES");
  PetscCall(SNESGetDM((SNES) ce->solver, &ce->idm));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstInitGuessSNES_Private(PetscConvEst ce, PetscInt r, DM dm, Vec u)
{
  PetscFunctionBegin;
  PetscCall(DMProjectFunction(dm, 0.0, ce->initGuess, ce->ctxs, INSERT_VALUES, u));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstComputeErrorSNES_Private(PetscConvEst ce, PetscInt r, DM dm, Vec u, PetscReal errors[])
{
  PetscFunctionBegin;
  PetscCall(DMComputeL2FieldDiff(dm, 0.0, ce->exactSol, ce->ctxs, u, errors));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstSetJacobianNullspace_Private(PetscConvEst ce, SNES snes)
{
  DM             dm;
  PetscInt       f;

  PetscFunctionBegin;
  PetscCall(SNESGetDM(snes, &dm));
  for (f = 0; f < ce->Nf; ++f) {
    PetscErrorCode (*nspconstr)(DM, PetscInt, PetscInt, MatNullSpace *);

    PetscCall(DMGetNullSpaceConstructor(dm, f, &nspconstr));
    if (nspconstr) {
      MatNullSpace nullsp;
      Mat          J;

      PetscCall((*nspconstr)(dm, f, f,&nullsp));
      PetscCall(SNESSetUp(snes));
      PetscCall(SNESGetJacobian(snes, &J, NULL, NULL, NULL));
      PetscCall(MatSetNullSpace(J, nullsp));
      PetscCall(MatNullSpaceDestroy(&nullsp));
      break;
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstGetConvRateSNES_Private(PetscConvEst ce, PetscReal alpha[])
{
  SNES           snes = (SNES) ce->solver;
  DM            *dm;
  PetscObject    disc;
  PetscReal     *x, *y, slope, intercept;
  PetscInt       Nr = ce->Nr, r, f, dim, oldlevel, oldnlev;
  void          *ctx;

  PetscFunctionBegin;
  PetscCheck(ce->r == 2.0,PetscObjectComm((PetscObject) ce), PETSC_ERR_SUP, "Only refinement factor 2 is currently supported (not %g)", (double) ce->r);
  PetscCall(DMGetDimension(ce->idm, &dim));
  PetscCall(DMGetApplicationContext(ce->idm, &ctx));
  PetscCall(DMPlexSetRefinementUniform(ce->idm, PETSC_TRUE));
  PetscCall(DMGetRefineLevel(ce->idm, &oldlevel));
  PetscCall(PetscMalloc1((Nr+1), &dm));
  /* Loop over meshes */
  dm[0] = ce->idm;
  for (r = 0; r <= Nr; ++r) {
    Vec           u;
#if defined(PETSC_USE_LOG)
    PetscLogStage stage;
#endif
    char          stageName[PETSC_MAX_PATH_LEN];
    const char   *dmname, *uname;

    PetscCall(PetscSNPrintf(stageName, PETSC_MAX_PATH_LEN-1, "ConvEst Refinement Level %" PetscInt_FMT, r));
#if defined(PETSC_USE_LOG)
    PetscCall(PetscLogStageGetId(stageName, &stage));
    if (stage < 0) PetscCall(PetscLogStageRegister(stageName, &stage));
#endif
    PetscCall(PetscLogStagePush(stage));
    if (r > 0) {
      if (!ce->noRefine) {
        PetscCall(DMRefine(dm[r-1], MPI_COMM_NULL, &dm[r]));
        PetscCall(DMSetCoarseDM(dm[r], dm[r-1]));
      } else {
        DM cdm, rcdm;

        PetscCall(DMClone(dm[r-1], &dm[r]));
        PetscCall(DMCopyDisc(dm[r-1], dm[r]));
        PetscCall(DMGetCoordinateDM(dm[r-1], &cdm));
        PetscCall(DMGetCoordinateDM(dm[r],   &rcdm));
        PetscCall(DMCopyDisc(cdm, rcdm));
      }
      PetscCall(DMCopyTransform(ce->idm, dm[r]));
      PetscCall(PetscObjectGetName((PetscObject) dm[r-1], &dmname));
      PetscCall(PetscObjectSetName((PetscObject) dm[r], dmname));
      for (f = 0; f < ce->Nf; ++f) {
        PetscErrorCode (*nspconstr)(DM, PetscInt, PetscInt, MatNullSpace *);

        PetscCall(DMGetNullSpaceConstructor(dm[r-1], f, &nspconstr));
        PetscCall(DMSetNullSpaceConstructor(dm[r],   f,  nspconstr));
      }
    }
    PetscCall(DMViewFromOptions(dm[r], NULL, "-conv_dm_view"));
    /* Create solution */
    PetscCall(DMCreateGlobalVector(dm[r], &u));
    PetscCall(DMGetField(dm[r], 0, NULL, &disc));
    PetscCall(PetscObjectGetName(disc, &uname));
    PetscCall(PetscObjectSetName((PetscObject) u, uname));
    /* Setup solver */
    PetscCall(SNESReset(snes));
    PetscCall(SNESSetDM(snes, dm[r]));
    PetscCall(DMPlexSetSNESLocalFEM(dm[r], ctx, ctx, ctx));
    PetscCall(SNESSetFromOptions(snes));
    /* Set nullspace for Jacobian */
    PetscCall(PetscConvEstSetJacobianNullspace_Private(ce, snes));
    /* Create initial guess */
    PetscCall(PetscConvEstComputeInitialGuess(ce, r, dm[r], u));
    PetscCall(SNESSolve(snes, NULL, u));
    PetscCall(PetscLogEventBegin(ce->event, ce, 0, 0, 0));
    PetscCall(PetscConvEstComputeError(ce, r, dm[r], u, &ce->errors[r*ce->Nf]));
    PetscCall(PetscLogEventEnd(ce->event, ce, 0, 0, 0));
    for (f = 0; f < ce->Nf; ++f) {
      PetscSection s, fs;
      PetscInt     lsize;

      /* Could use DMGetOutputDM() to add in Dirichlet dofs */
      PetscCall(DMGetLocalSection(dm[r], &s));
      PetscCall(PetscSectionGetField(s, f, &fs));
      PetscCall(PetscSectionGetConstrainedStorageSize(fs, &lsize));
      PetscCallMPI(MPI_Allreduce(&lsize, &ce->dofs[r*ce->Nf+f], 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) snes)));
      PetscCall(PetscLogEventSetDof(ce->event, f, ce->dofs[r*ce->Nf+f]));
      PetscCall(PetscLogEventSetError(ce->event, f, ce->errors[r*ce->Nf+f]));
    }
    /* Monitor */
    PetscCall(PetscConvEstMonitorDefault(ce, r));
    if (!r) {
      /* PCReset() does not wipe out the level structure */
      KSP ksp;
      PC  pc;

      PetscCall(SNESGetKSP(snes, &ksp));
      PetscCall(KSPGetPC(ksp, &pc));
      PetscCall(PCMGGetLevels(pc, &oldnlev));
    }
    /* Cleanup */
    PetscCall(VecDestroy(&u));
    PetscCall(PetscLogStagePop());
  }
  for (r = 1; r <= Nr; ++r) {
    PetscCall(DMDestroy(&dm[r]));
  }
  /* Fit convergence rate */
  PetscCall(PetscMalloc2(Nr+1, &x, Nr+1, &y));
  for (f = 0; f < ce->Nf; ++f) {
    for (r = 0; r <= Nr; ++r) {
      x[r] = PetscLog10Real(ce->dofs[r*ce->Nf+f]);
      y[r] = PetscLog10Real(ce->errors[r*ce->Nf+f]);
    }
    PetscCall(PetscLinearRegression(Nr+1, x, y, &slope, &intercept));
    /* Since h^{-dim} = N, lg err = s lg N + b = -s dim lg h + b */
    alpha[f] = -slope * dim;
  }
  PetscCall(PetscFree2(x, y));
  PetscCall(PetscFree(dm));
  /* Restore solver */
  PetscCall(SNESReset(snes));
  {
    /* PCReset() does not wipe out the level structure */
    KSP ksp;
    PC  pc;

    PetscCall(SNESGetKSP(snes, &ksp));
    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PCMGSetLevels(pc, oldnlev, NULL));
    PetscCall(DMSetRefineLevel(ce->idm, oldlevel)); /* The damn DMCoarsen() calls in PCMG can reset this */
  }
  PetscCall(SNESSetDM(snes, ce->idm));
  PetscCall(DMPlexSetSNESLocalFEM(ce->idm, ctx, ctx, ctx));
  PetscCall(SNESSetFromOptions(snes));
  PetscCall(PetscConvEstSetJacobianNullspace_Private(ce, snes));
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstGetConvRate - Returns an estimate of the convergence rate for the discretization

  Not collective

  Input Parameter:
. ce   - The PetscConvEst object

  Output Parameter:
. alpha - The convergence rate for each field

  Note: The convergence rate alpha is defined by
$ || u_\Delta - u_exact || < C \Delta^alpha
where u_\Delta is the discrete solution, and Delta is a measure of the discretization size. We usually use h for the
spatial resolution and \Delta t for the temporal resolution.

We solve a series of problems using increasing resolution (refined meshes or decreased timesteps), calculate an error
based upon the exact solution in the DS, and then fit the result to our model above using linear regression.

  Options database keys:
+ -snes_convergence_estimate - Execute convergence estimation inside SNESSolve() and print out the rate
- -ts_convergence_estimate - Execute convergence estimation inside TSSolve() and print out the rate

  Level: intermediate

.seealso: `PetscConvEstSetSolver()`, `PetscConvEstCreate()`, `PetscConvEstGetConvRate()`, `SNESSolve()`, `TSSolve()`
@*/
PetscErrorCode PetscConvEstGetConvRate(PetscConvEst ce, PetscReal alpha[])
{
  PetscInt       f;

  PetscFunctionBegin;
  if (ce->event < 0) PetscCall(PetscLogEventRegister("ConvEst Error", PETSC_OBJECT_CLASSID, &ce->event));
  for (f = 0; f < ce->Nf; ++f) alpha[f] = 0.0;
  PetscCall((*ce->ops->getconvrate)(ce, alpha));
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstRateView - Displays the convergence rate to a viewer

   Collective on SNES

   Parameter:
+  snes - iterative context obtained from SNESCreate()
.  alpha - the convergence rate for each field
-  viewer - the viewer to display the reason

   Options Database Keys:
.  -snes_convergence_estimate - print the convergence rate

   Level: developer

.seealso: `PetscConvEstGetRate()`
@*/
PetscErrorCode PetscConvEstRateView(PetscConvEst ce, const PetscReal alpha[], PetscViewer viewer)
{
  PetscBool      isAscii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isAscii));
  if (isAscii) {
    PetscInt Nf = ce->Nf, f;

    PetscCall(PetscViewerASCIIAddTab(viewer, ((PetscObject) ce)->tablevel));
    PetscCall(PetscViewerASCIIPrintf(viewer, "L_2 convergence rate: "));
    if (Nf > 1) PetscCall(PetscViewerASCIIPrintf(viewer, "["));
    for (f = 0; f < Nf; ++f) {
      if (f > 0) PetscCall(PetscViewerASCIIPrintf(viewer, ", "));
      PetscCall(PetscViewerASCIIPrintf(viewer, "%#.2g", (double) alpha[f]));
    }
    if (Nf > 1) PetscCall(PetscViewerASCIIPrintf(viewer, "]"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "\n"));
    PetscCall(PetscViewerASCIISubtractTab(viewer, ((PetscObject) ce)->tablevel));
  }
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstCreate - Create a PetscConvEst object

  Collective

  Input Parameter:
. comm - The communicator for the PetscConvEst object

  Output Parameter:
. ce   - The PetscConvEst object

  Level: beginner

.seealso: `PetscConvEstDestroy()`, `PetscConvEstGetConvRate()`
@*/
PetscErrorCode PetscConvEstCreate(MPI_Comm comm, PetscConvEst *ce)
{
  PetscFunctionBegin;
  PetscValidPointer(ce, 2);
  PetscCall(PetscSysInitializePackage());
  PetscCall(PetscHeaderCreate(*ce, PETSC_OBJECT_CLASSID, "PetscConvEst", "ConvergenceEstimator", "SNES", comm, PetscConvEstDestroy, PetscConvEstView));
  (*ce)->monitor = PETSC_FALSE;
  (*ce)->r       = 2.0;
  (*ce)->Nr      = 4;
  (*ce)->event   = -1;
  (*ce)->ops->setsolver    = PetscConvEstSetSNES_Private;
  (*ce)->ops->initguess    = PetscConvEstInitGuessSNES_Private;
  (*ce)->ops->computeerror = PetscConvEstComputeErrorSNES_Private;
  (*ce)->ops->getconvrate  = PetscConvEstGetConvRateSNES_Private;
  PetscFunctionReturn(0);
}
