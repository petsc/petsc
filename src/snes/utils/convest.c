#include <petscconvest.h>            /*I "petscconvest.h" I*/
#include <petscdmplex.h>
#include <petscds.h>
#include <petscblaslapack.h>

#include <petsc/private/petscconvestimpl.h>

static PetscErrorCode zero_private(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nc, PetscScalar *u, void *ctx)
{
  PetscInt c;
  for (c = 0; c < Nc; ++c) u[c] = 0.0;
  return 0;
}


/*@
  PetscConvEstCreate - Create a PetscConvEst object

  Collective on MPI_Comm

  Input Parameter:
. comm - The communicator for the PetscConvEst object

  Output Parameter:
. ce   - The PetscConvEst object

  Level: beginner

.keywords: PetscConvEst, convergence, create
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

.keywords: PetscConvEst, convergence, destroy
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
  ierr = PetscFree2((*ce)->initGuess, (*ce)->exactSol);CHKERRQ(ierr);
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

.keywords: PetscConvEst, convergence, options
.seealso: PetscConvEstCreate(), PetscConvEstGetConvRate()
@*/
PetscErrorCode PetscConvEstSetFromOptions(PetscConvEst ce)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsBegin(PetscObjectComm((PetscObject) ce), "", "Convergence Estimator Options", "PetscConvEst");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-num_refine", "The number of refinements for the convergence check", "PetscConvEst", ce->Nr, &ce->Nr, NULL);CHKERRQ(ierr);
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

.keywords: PetscConvEst, convergence, view
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

.keywords: PetscConvEst, convergence
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

.keywords: PetscConvEst, convergence
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

.keywords: PetscConvEst, convergence, setup
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
  ierr = PetscMalloc2(ce->Nf, &ce->initGuess, ce->Nf, &ce->exactSol);CHKERRQ(ierr);
  for (f = 0; f < ce->Nf; ++f) ce->initGuess[f] = zero_private;
  for (f = 0; f < ce->Nf; ++f) {
    ierr = PetscDSGetExactSolution(prob, f, &ce->exactSol[f]);CHKERRQ(ierr);
    if (!ce->exactSol[f]) SETERRQ1(PetscObjectComm((PetscObject) ce), PETSC_ERR_ARG_WRONG, "DS must contain exact solution functions in order to estimate convergence, missing for field %D", f);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscConvEstLinearRegression_Private(PetscConvEst ce, PetscInt n, const PetscReal x[], const PetscReal y[], PetscReal *slope, PetscReal *intercept)
{
  PetscScalar    H[4];
  PetscReal     *X, *Y, beta[2];
  PetscInt       i, j, k;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  *slope = *intercept = 0.0;
  ierr = PetscMalloc2(n*2, &X, n*2, &Y);CHKERRQ(ierr);
  for (k = 0; k < n; ++k) {
    /* X[n,2] = [1, x] */
    X[k*2+0] = 1.0;
    X[k*2+1] = x[k];
  }
  /* H = X^T X */
  for (i = 0; i < 2; ++i) {
    for (j = 0; j < 2; ++j) {
      H[i*2+j] = 0.0;
      for (k = 0; k < n; ++k) {
        H[i*2+j] += X[k*2+i] * X[k*2+j];
      }
    }
  }
  /* H = (X^T X)^{-1} */
  {
    PetscBLASInt two = 2, ipiv[2], info;
    PetscScalar  work[2];

    ierr = PetscFPTrapPush(PETSC_FP_TRAP_OFF);CHKERRQ(ierr);
    PetscStackCallBLAS("LAPACKgetrf", LAPACKgetrf_(&two, &two, H, &two, ipiv, &info));
    PetscStackCallBLAS("LAPACKgetri", LAPACKgetri_(&two, H, &two, ipiv, work, &two, &info));
    ierr = PetscFPTrapPop();CHKERRQ(ierr);
  }
    /* Y = H X^T */
  for (i = 0; i < 2; ++i) {
    for (k = 0; k < n; ++k) {
      Y[i*n+k] = 0.0;
      for (j = 0; j < 2; ++j) {
        Y[i*n+k] += PetscRealPart(H[i*2+j]) * X[k*2+j];
      }
    }
  }
  /* beta = Y error = [y-intercept, slope] */
  for (i = 0; i < 2; ++i) {
    beta[i] = 0.0;
    for (k = 0; k < n; ++k) {
      beta[i] += Y[i*n+k] * y[k];
    }
  }
  ierr = PetscFree2(X, Y);CHKERRQ(ierr);
  *intercept = beta[0];
  *slope     = beta[1];
  PetscFunctionReturn(0);
}

/*@
  PetscConvEstGetConvRate - Returns an estimate of the convergence rate for the discretization

  Not collective

  Input Parameter:
. ce   - The PetscConvEst object

  Output Parameter:
. alpha - The convergence rate

  Note: The convergence rate alpha is defined by
$ || u_h - u_exact || < C h^alpha
where u_h is the discrete solution, and h is a measure of the discretization size.

We solve a series of problems on refined meshes, calculate an error based upon the exact solution in the DS,
and then fit the result to our model above using linear regression.

  Options database keys:
. -snes_convergence_estimate : Execute convergence estimation and print out the rate

  Level: intermediate

.keywords: PetscConvEst, convergence
.seealso: PetscConvEstSetSolver(), PetscConvEstCreate(), PetscConvEstGetConvRate()
@*/
PetscErrorCode PetscConvEstGetConvRate(PetscConvEst ce, PetscReal *alpha)
{
  DM            *dm;
  PetscDS        prob;
  PetscObject    disc;
  MPI_Comm       comm;
  const char    *uname, *dmname;
  void          *ctx;
  Vec            u;
  PetscReal      t = 0.0, *x, *y, slope, intercept;
  PetscInt      *dof, dim, Nr = ce->Nr, r;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) ce, &comm);CHKERRQ(ierr);
  ierr = DMGetDimension(ce->idm, &dim);CHKERRQ(ierr);
  ierr = DMGetApplicationContext(ce->idm, &ctx);CHKERRQ(ierr);
  ierr = DMGetDS(ce->idm, &prob);CHKERRQ(ierr);
  ierr = DMPlexSetRefinementUniform(ce->idm, PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscMalloc2((Nr+1), &dm, (Nr+1), &dof);CHKERRQ(ierr);
  dm[0]  = ce->idm;
  *alpha = 0.0;
  /* Loop over meshes */
  for (r = 0; r <= Nr; ++r) {
    if (r > 0) {
      ierr = DMRefine(dm[r-1], MPI_COMM_NULL, &dm[r]);CHKERRQ(ierr);
      ierr = DMSetCoarseDM(dm[r], dm[r-1]);CHKERRQ(ierr);
      ierr = DMSetDS(dm[r], prob);CHKERRQ(ierr);
      ierr = PetscObjectGetName((PetscObject) dm[r-1], &dmname);CHKERRQ(ierr);
      ierr = PetscObjectSetName((PetscObject) dm[r], dmname);CHKERRQ(ierr);
    }
    ierr = DMViewFromOptions(dm[r], NULL, "-conv_dm_view");CHKERRQ(ierr);
    ierr = DMPlexGetHeightStratum(dm[r], 0, NULL, &dof[r]);CHKERRQ(ierr);
    /* Create solution */
    ierr = DMCreateGlobalVector(dm[r], &u);CHKERRQ(ierr);
    ierr = PetscDSGetDiscretization(prob, 0, &disc);CHKERRQ(ierr);
    ierr = PetscObjectGetName(disc, &uname);CHKERRQ(ierr);
    ierr = PetscObjectSetName((PetscObject) u, uname);CHKERRQ(ierr);
    /* Setup solver */
    ierr = SNESReset(ce->snes);CHKERRQ(ierr);
    ierr = SNESSetDM(ce->snes, dm[r]);CHKERRQ(ierr);
    ierr = DMPlexSetSNESLocalFEM(dm[r], ctx, ctx, ctx);CHKERRQ(ierr);
    ierr = SNESSetFromOptions(ce->snes);CHKERRQ(ierr);
    /* Create initial guess */
    ierr = DMProjectFunction(dm[r], t, ce->initGuess, NULL, INSERT_VALUES, u);CHKERRQ(ierr);
    ierr = SNESSolve(ce->snes, NULL, u);CHKERRQ(ierr);
    ierr = DMComputeL2FieldDiff(dm[r], t, ce->exactSol, NULL, u, &ce->errors[r*ce->Nf]);CHKERRQ(ierr);
    /* Monitor */
    if (ce->monitor) {
      PetscReal *errors = &ce->errors[r*ce->Nf];
      PetscInt f;

      ierr = PetscPrintf(comm, "L_2 Error: [");CHKERRQ(ierr);
      for (f = 0; f < ce->Nf; ++f) {
        if (f > 0) {ierr = PetscPrintf(comm, ", ");CHKERRQ(ierr);}
        if (errors[f] < 1.0e-11) {ierr = PetscPrintf(comm, "< 1e-11");CHKERRQ(ierr);}
        else                     {ierr = PetscPrintf(comm, "%g", (double)errors[f]);CHKERRQ(ierr);}
      }
      ierr = PetscPrintf(comm, "]\n");CHKERRQ(ierr);
    }
    /* Cleanup */
    ierr = VecDestroy(&u);CHKERRQ(ierr);
  }
  for (r = 1; r <= Nr; ++r) {
    ierr = DMDestroy(&dm[r]);CHKERRQ(ierr);
  }
  /* Fit convergence rate */
  ierr = PetscMalloc2(Nr+1, &x, Nr+1, &y);CHKERRQ(ierr);
  for (r = 0; r <= Nr; ++r) {
    x[r] = PetscLog10Real(dof[r]);
    y[r] = PetscLog10Real(ce->errors[r*ce->Nf+0]);
  }
  ierr = PetscConvEstLinearRegression_Private(ce, Nr+1, x, y, &slope, &intercept);CHKERRQ(ierr);
  ierr = PetscFree2(x, y);CHKERRQ(ierr);
  /* Since h^{-dim} = N, lg err = s lg N + b = -s dim lg h + b */
  *alpha = -slope * dim;
  ierr = PetscFree2(dm, dof);CHKERRQ(ierr);
  /* Restore solver */
  ierr = SNESReset(ce->snes);CHKERRQ(ierr);
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
.  alpha - the convergence rate
-  viewer - the viewer to display the reason

   Options Database Keys:
.  -snes_convergence_estimate - print the convergence rate

   Level: developer

.seealso: PetscConvEstGetRate()
@*/
PetscErrorCode PetscConvEstRateView(PetscConvEst ce, PetscReal alpha, PetscViewer viewer)
{
  PetscBool      isAscii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isAscii);CHKERRQ(ierr);
  if (isAscii) {
    ierr = PetscViewerASCIIAddTab(viewer, ((PetscObject) ce)->tablevel);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "L_2 convergence rate: %g\n", (double) alpha);CHKERRQ(ierr);
    ierr = PetscViewerASCIISubtractTab(viewer, ((PetscObject) ce)->tablevel);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}
