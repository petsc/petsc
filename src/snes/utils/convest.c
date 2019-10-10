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
  PetscConvEstCreate - Create a PetscConvEst object

  Collective

  Input Parameter:
. comm - The communicator for the PetscConvEst object

  Output Parameter:
. ce   - The PetscConvEst object

  Level: beginner

.seealso: PetscConvEstDestroy(), PetscConvEstGetConvRate()
@*/
PetscErrorCode PetscConvEstCreate(MPI_Comm comm, PetscConvEst *ce)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidPointer(ce, 2);
  ierr = PetscSysInitializePackage();CHKERRQ(ierr);
  ierr = PetscHeaderCreate(*ce, PETSC_OBJECT_CLASSID, "PetscConvEst", "ConvergenceEstimator", "SNES", comm, PetscConvEstDestroy, PetscConvEstView);CHKERRQ(ierr);
  (*ce)->monitor = PETSC_FALSE;
  (*ce)->Nr      = 4;
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstDestroy - Destroys a PetscConvEst object

  Collective on PetscConvEst

  Input Parameter:
. ce - The PetscConvEst object

  Level: beginner

.seealso: PetscConvEstCreate(), PetscConvEstGetConvRate()
@*/
PetscErrorCode PetscConvEstDestroy(PetscConvEst *ce)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*ce) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*ce),PETSC_OBJECT_CLASSID,1);
  if (--((PetscObject)(*ce))->refct > 0) {
    *ce = NULL;
    PetscFunctionReturn(0);
  }
  ierr = PetscFree3((*ce)->initGuess, (*ce)->exactSol, (*ce)->ctxs);CHKERRQ(ierr);
  ierr = PetscFree((*ce)->errors);CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(ce);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstSetFromOptions - Sets a PetscConvEst object from options

  Collective on PetscConvEst

  Input Parameters:
. ce - The PetscConvEst object

  Level: beginner

.seealso: PetscConvEstCreate(), PetscConvEstGetConvRate()
@*/
PetscErrorCode PetscConvEstSetFromOptions(PetscConvEst ce)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject) ce), "", "Convergence Estimator Options", "PetscConvEst");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-convest_num_refine", "The number of refinements for the convergence check", "PetscConvEst", ce->Nr, &ce->Nr, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-convest_monitor", "Monitor the error for each convergence check", "PetscConvEst", ce->monitor, &ce->monitor, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstView - Views a PetscConvEst object

  Collective on PetscConvEst

  Input Parameters:
+ ce     - The PetscConvEst object
- viewer - The PetscViewer object

  Level: beginner

.seealso: PetscConvEstCreate(), PetscConvEstGetConvRate()
@*/
PetscErrorCode PetscConvEstView(PetscConvEst ce, PetscViewer viewer)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectPrintClassNamePrefixType((PetscObject) ce, viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPrintf(viewer, "ConvEst with %D levels\n", ce->Nr+1);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstGetSolver - Gets the solver used to produce discrete solutions

  Not collective

  Input Parameter:
. ce   - The PetscConvEst object

  Output Parameter:
. snes - The solver

  Level: intermediate

.seealso: PetscConvEstSetSolver(), PetscConvEstCreate(), PetscConvEstGetConvRate()
@*/
PetscErrorCode PetscConvEstGetSolver(PetscConvEst ce, SNES *snes)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(ce, PETSC_OBJECT_CLASSID, 1);
  PetscValidPointer(snes, 2);
  *snes = ce->snes;
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstSetSolver - Sets the solver used to produce discrete solutions

  Not collective

  Input Parameters:
+ ce   - The PetscConvEst object
- snes - The solver

  Level: intermediate

  Note: The solver MUST have an attached DM/DS, so that we know the exact solution

.seealso: PetscConvEstGetSolver(), PetscConvEstCreate(), PetscConvEstGetConvRate()
@*/
PetscErrorCode PetscConvEstSetSolver(PetscConvEst ce, SNES snes)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ce, PETSC_OBJECT_CLASSID, 1);
  PetscValidHeaderSpecific(snes, SNES_CLASSID, 2);
  ce->snes = snes;
  ierr = SNESGetDM(ce->snes, &ce->idm);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstSetUp - After the solver is specified, we create structures for estimating convergence

  Collective on PetscConvEst

  Input Parameters:
. ce - The PetscConvEst object

  Level: beginner

.seealso: PetscConvEstCreate(), PetscConvEstGetConvRate()
@*/
PetscErrorCode PetscConvEstSetUp(PetscConvEst ce)
{
  PetscDS        prob;
  PetscInt       f;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMGetDS(ce->idm, &prob);CHKERRQ(ierr);
  ierr = PetscDSGetNumFields(prob, &ce->Nf);CHKERRQ(ierr);
  ierr = PetscMalloc1((ce->Nr+1)*ce->Nf, &ce->errors);CHKERRQ(ierr);
  ierr = PetscMalloc3(ce->Nf, &ce->initGuess, ce->Nf, &ce->exactSol, ce->Nf, &ce->ctxs);CHKERRQ(ierr);
  for (f = 0; f < ce->Nf; ++f) ce->initGuess[f] = zero_private;
  for (f = 0; f < ce->Nf; ++f) {
    ierr = PetscDSGetExactSolution(prob, f, &ce->exactSol[f], &ce->ctxs[f]);CHKERRQ(ierr);
    if (!ce->exactSol[f]) SETERRQ1(PetscObjectComm((PetscObject) ce), PETSC_ERR_ARG_WRONG, "DS must contain exact solution functions in order to estimate convergence, missing for field %D", f);
  }
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
$ || u_h - u_exact || < C h^alpha
where u_h is the discrete solution, and h is a measure of the discretization size.

We solve a series of problems on refined meshes, calculate an error based upon the exact solution in the DS,
and then fit the result to our model above using linear regression.

  Options database keys:
. -snes_convergence_estimate : Execute convergence estimation and print out the rate

  Level: intermediate

.seealso: PetscConvEstSetSolver(), PetscConvEstCreate(), PetscConvEstGetConvRate()
@*/
PetscErrorCode PetscConvEstGetConvRate(PetscConvEst ce, PetscReal alpha[])
{
  DM            *dm;
  PetscObject    disc;
  MPI_Comm       comm;
  const char    *uname, *dmname;
  void          *ctx;
  Vec            u;
  PetscReal      t = 0.0, *x, *y, slope, intercept;
  PetscInt      *dof, dim, Nr = ce->Nr, r, f, oldlevel, oldnlev;
  PetscLogEvent  event;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) ce, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(ce->idm, &dim);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(ce->idm, &ctx);CHKERRQ(ierr);
  ierr = DMPlexSetRefinementUniform(ce->idm, PETSC_TRUE);CHKERRQ(ierr);
  ierr = DMGetRefineLevel(ce->idm, &oldlevel);CHKERRQ(ierr);
  ierr = PetscMalloc2((Nr+1), &dm, (Nr+1)*ce->Nf, &dof);CHKERRQ(ierr);
  dm[0]  = ce->idm;
  for (f = 0; f < ce->Nf; ++f) alpha[f] = 0.0;
  /* Loop over meshes */
  ierr = PetscLogEventRegister("ConvEst Error", PETSC_OBJECT_CLASSID, &event);CHKERRQ(ierr);
  for (r = 0; r <= Nr; ++r) {
    PetscLogStage stage;
    char          stageName[PETSC_MAX_PATH_LEN];

    ierr = PetscSNPrintf(stageName, PETSC_MAX_PATH_LEN-1, "ConvEst Refinement Level %D", r);CHKERRQ(ierr);
    ierr = PetscLogStageRegister(stageName, &stage);CHKERRQ(ierr);
    ierr = PetscLogStagePush(stage);CHKERRQ(ierr);
    if (r > 0) {
      ierr = DMRefine(dm[r-1], MPI_COMM_NULL, &dm[r]);CHKERRQ(ierr);
      ierr = DMSetCoarseDM(dm[r], dm[r-1]);CHKERRQ(ierr);
      ierr = DMCopyDisc(ce->idm, dm[r]);CHKERRQ(ierr);
      ierr = DMCopyTransform(ce->idm, dm[r]);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) dm[r-1], &dmname);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) dm[r], dmname);CHKERRQ(ierr);
      for (f = 0; f <= ce->Nf; ++f) {
        PetscErrorCode (*nspconstr)(DM, PetscInt, MatNullSpace *);
        ierr = DMGetNullSpaceConstructor(dm[r-1], f, &nspconstr);CHKERRQ(ierr);
        ierr = DMSetNullSpaceConstructor(dm[r],   f,  nspconstr);CHKERRQ(ierr);
      }
    }
    ierr = DMViewFromOptions(dm[r], NULL, "-conv_dm_view");CHKERRQ(ierr);
    /* Create solution */
    ierr = DMCreateGlobalVector(dm[r], &u);CHKERRQ(ierr);
    ierr = DMGetField(dm[r], 0, NULL, &disc);CHKERRQ(ierr);
    ierr = PetscObjectGetName(disc, &uname);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, uname);CHKERRQ(ierr);
    /* Setup solver */
    ierr = SNESReset(ce->snes);CHKERRQ(ierr);
    ierr = SNESSetDM(ce->snes, dm[r]);CHKERRQ(ierr);
    ierr = DMPlexSetSNESLocalFEM(dm[r], ctx, ctx, ctx);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(ce->snes);CHKERRQ(ierr);
    /* Create initial guess */
    ierr = DMProjectFunction(dm[r], t, ce->initGuess, ce->ctxs, INSERT_VALUES, u);CHKERRQ(ierr);
    ierr = SNESSolve(ce->snes, NULL, u);CHKERRQ(ierr);
    ierr = PetscLogEventBegin(event, ce, 0, 0, 0);CHKERRQ(ierr);
    ierr = DMComputeL2FieldDiff(dm[r], t, ce->exactSol, ce->ctxs, u, &ce->errors[r*ce->Nf]);CHKERRQ(ierr);
    ierr = PetscLogEventEnd(event, ce, 0, 0, 0);CHKERRQ(ierr);
    for (f = 0; f < ce->Nf; ++f) {
      PetscSection s, fs;
      PetscInt     lsize;

      /* Could use DMGetOutputDM() to add in Dirichlet dofs */
      ierr = DMGetLocalSection(dm[r], &s);CHKERRQ(ierr);
      ierr = PetscSectionGetField(s, f, &fs);CHKERRQ(ierr);
      ierr = PetscSectionGetConstrainedStorageSize(fs, &lsize);CHKERRQ(ierr);
      ierr = MPI_Allreduce(&lsize, &dof[r*ce->Nf+f], 1, MPIU_INT, MPI_SUM, PetscObjectComm((PetscObject) ce->snes));CHKERRQ(ierr);
      ierr = PetscLogEventSetDof(event, f, dof[r*ce->Nf+f]);CHKERRQ(ierr);
      ierr = PetscLogEventSetError(event, f, ce->errors[r*ce->Nf+f]);CHKERRQ(ierr);
    }
    /* Monitor */
    if (ce->monitor) {
      PetscReal *errors = &ce->errors[r*ce->Nf];

      ierr = PetscPrintf(comm, "L_2 Error: ");CHKERRQ(ierr);
      if (ce->Nf > 1) {ierr = PetscPrintf(comm, "[");CHKERRQ(ierr);}
      for (f = 0; f < ce->Nf; ++f) {
        if (f > 0) {ierr = PetscPrintf(comm, ", ");CHKERRQ(ierr);}
        if (errors[f] < 1.0e-11) {ierr = PetscPrintf(comm, "< 1e-11");CHKERRQ(ierr);}
        else                     {ierr = PetscPrintf(comm, "%g", (double)errors[f]);CHKERRQ(ierr);}
      }
      if (ce->Nf > 1) {ierr = PetscPrintf(comm, "]");CHKERRQ(ierr);}
      ierr = PetscPrintf(comm, "\n");CHKERRQ(ierr);
    }
    if (!r) {
      /* PCReset() does not wipe out the level structure */
      KSP ksp;
      PC  pc;

      ierr = SNESGetKSP(ce->snes, &ksp);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
      ierr = PCMGGetLevels(pc, &oldnlev);CHKERRQ(ierr);
    }
    /* Cleanup */
    ierr = VecDestroy(&u);CHKERRQ(ierr);
    ierr = PetscLogStagePop();CHKERRQ(ierr);
  }
  for (r = 1; r <= Nr; ++r) {
    ierr = DMDestroy(&dm[r]);CHKERRQ(ierr);
  }
  /* Fit convergence rate */
  ierr = PetscMalloc2(Nr+1, &x, Nr+1, &y);CHKERRQ(ierr);
  for (f = 0; f < ce->Nf; ++f) {
    for (r = 0; r <= Nr; ++r) {
      x[r] = PetscLog10Real(dof[r*ce->Nf+f]);
      y[r] = PetscLog10Real(ce->errors[r*ce->Nf+f]);
    }
    ierr = PetscLinearRegression(Nr+1, x, y, &slope, &intercept);CHKERRQ(ierr);
    /* Since h^{-dim} = N, lg err = s lg N + b = -s dim lg h + b */
    alpha[f] = -slope * dim;
  }
  ierr = PetscFree2(x, y);CHKERRQ(ierr);
  ierr = PetscFree2(dm, dof);CHKERRQ(ierr);
  /* Restore solver */
  ierr = SNESReset(ce->snes);CHKERRQ(ierr);
  {
    /* PCReset() does not wipe out the level structure */
    KSP ksp;
    PC  pc;

    ierr = SNESGetKSP(ce->snes, &ksp);CHKERRQ(ierr);
    ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
    ierr = PCMGSetLevels(pc, oldnlev, NULL);CHKERRQ(ierr);
    ierr = DMSetRefineLevel(ce->idm, oldlevel);CHKERRQ(ierr); /* The damn DMCoarsen() calls in PCMG can reset this */
  }
  ierr = SNESSetDM(ce->snes, ce->idm);CHKERRQ(ierr);
  ierr = DMPlexSetSNESLocalFEM(ce->idm, ctx, ctx, ctx);CHKERRQ(ierr);
  ierr = SNESSetFromOptions(ce->snes);CHKERRQ(ierr);
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

.seealso: PetscConvEstGetRate()
@*/
PetscErrorCode PetscConvEstRateView(PetscConvEst ce, const PetscReal alpha[], PetscViewer viewer)
{
  PetscBool      isAscii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isAscii);CHKERRQ(ierr);
  if (isAscii) {
    DM       dm;
    PetscInt Nf, f;

    ierr = SNESGetDM(ce->snes, &dm);CHKERRQ(ierr);
    ierr = DMGetNumFields(dm, &Nf);CHKERRQ(ierr);
    ierr = PetscViewerASCIIAddTab(viewer, ((PetscObject) ce)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "L_2 convergence rate: ");CHKERRQ(ierr);
    if (Nf > 1) {ierr = PetscViewerASCIIPrintf(viewer, "[");CHKERRQ(ierr);}
    for (f = 0; f < Nf; ++f) {
      if (f > 0) {ierr = PetscViewerASCIIPrintf(viewer, ", ");CHKERRQ(ierr);}
      ierr = PetscViewerASCIIPrintf(viewer, "%#.2g", (double) alpha[f]);CHKERRQ(ierr);
    }
    if (Nf > 1) {ierr = PetscViewerASCIIPrintf(viewer, "]");CHKERRQ(ierr);}
    ierr = PetscViewerASCIIPrintf(viewer, "\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer, ((PetscObject) ce)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
