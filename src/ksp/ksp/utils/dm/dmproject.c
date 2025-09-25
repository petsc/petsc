#include <petsc/private/dmimpl.h>
#include <petscdm.h>      /*I "petscdm.h" I*/
#include <petscdmda.h>    /*I "petscdmda.h" I*/
#include <petscdmplex.h>  /*I "petscdmplex.h" I*/
#include <petscdmswarm.h> /*I "petscdmswarm.h" I*/
#include <petscksp.h>     /*I "petscksp.h" I*/
#include <petscblaslapack.h>

#include <petsc/private/dmswarmimpl.h>         // For the citation and check
#include "../src/dm/impls/swarm/data_bucket.h" // For DataBucket internals
#include "petscmath.h"

typedef struct _projectConstraintsCtx {
  DM  dm;
  Vec mask;
} projectConstraintsCtx;

static PetscErrorCode MatMult_GlobalToLocalNormal(Mat CtC, Vec x, Vec y)
{
  DM                     dm;
  Vec                    local, mask;
  projectConstraintsCtx *ctx;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(CtC, &ctx));
  dm   = ctx->dm;
  mask = ctx->mask;
  PetscCall(DMGetLocalVector(dm, &local));
  PetscCall(DMGlobalToLocalBegin(dm, x, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(dm, x, INSERT_VALUES, local));
  if (mask) PetscCall(VecPointwiseMult(local, mask, local));
  PetscCall(VecSet(y, 0.));
  PetscCall(DMLocalToGlobalBegin(dm, local, ADD_VALUES, y));
  PetscCall(DMLocalToGlobalEnd(dm, local, ADD_VALUES, y));
  PetscCall(DMRestoreLocalVector(dm, &local));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMGlobalToLocalSolve_project1(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar u[], void *ctx)
{
  PetscInt f;

  PetscFunctionBegin;
  for (f = 0; f < Nf; f++) u[f] = 1.;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMGlobalToLocalSolve - Solve for the global vector that is mapped to a given local vector by `DMGlobalToLocalBegin()`/`DMGlobalToLocalEnd()` with mode
  `INSERT_VALUES`.

  Collective

  Input Parameters:
+ dm - The `DM` object
. x  - The local vector
- y  - The global vector: the input value of this variable is used as an initial guess

  Output Parameter:
. y - The least-squares solution

  Level: advanced

  Note:
  It is assumed that the sum of all the local vector sizes is greater than or equal to the global vector size, so the solution is
  a least-squares solution.  It is also assumed that `DMLocalToGlobalBegin()`/`DMLocalToGlobalEnd()` with mode `ADD_VALUES` is the adjoint of the
  global-to-local map, so that the least-squares solution may be found by the normal equations.

  If the `DM` is of type `DMPLEX`, then `y` is the solution of $ L^T * D * L * y = L^T * D * x $, where $D$ is a diagonal mask that is 1 for every point in
  the union of the closures of the local cells and 0 otherwise.  This difference is only relevant if there are anchor points that are not in the
  closure of any local cell (see `DMPlexGetAnchors()`/`DMPlexSetAnchors()`).

  What is L?

  If this solves for a global vector from a local vector why is not called `DMLocalToGlobalSolve()`?

.seealso: [](ch_dmbase), `DM`, `DMGlobalToLocalBegin()`, `DMGlobalToLocalEnd()`, `DMLocalToGlobalBegin()`, `DMLocalToGlobalEnd()`, `DMPlexGetAnchors()`, `DMPlexSetAnchors()`
@*/
PetscErrorCode DMGlobalToLocalSolve(DM dm, Vec x, Vec y)
{
  Mat                   CtC;
  PetscInt              n, N, cStart, cEnd, c;
  PetscBool             isPlex;
  KSP                   ksp;
  PC                    pc;
  Vec                   global, mask = NULL;
  projectConstraintsCtx ctx;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isPlex));
  if (isPlex) {
    /* mark points in the closure */
    PetscCall(DMCreateLocalVector(dm, &mask));
    PetscCall(VecSet(mask, 0.0));
    PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
    if (cEnd > cStart) {
      PetscScalar *ones;
      PetscInt     numValues, i;

      PetscCall(DMPlexVecGetClosure(dm, NULL, mask, cStart, &numValues, NULL));
      PetscCall(PetscMalloc1(numValues, &ones));
      for (i = 0; i < numValues; i++) ones[i] = 1.;
      for (c = cStart; c < cEnd; c++) PetscCall(DMPlexVecSetClosure(dm, NULL, mask, c, ones, INSERT_VALUES));
      PetscCall(PetscFree(ones));
    }
  } else {
    PetscBool hasMask;

    PetscCall(DMHasNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &hasMask));
    if (!hasMask) {
      PetscErrorCode (**func)(PetscInt dim, PetscReal time, const PetscReal x[], PetscInt Nf, PetscScalar *u, void *ctx);
      void   **ctx;
      PetscInt Nf, f;

      PetscCall(DMGetNumFields(dm, &Nf));
      PetscCall(PetscMalloc2(Nf, &func, Nf, &ctx));
      for (f = 0; f < Nf; ++f) {
        func[f] = DMGlobalToLocalSolve_project1;
        ctx[f]  = NULL;
      }
      PetscCall(DMGetNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask));
      PetscCall(DMProjectFunctionLocal(dm, 0.0, func, ctx, INSERT_ALL_VALUES, mask));
      PetscCall(DMRestoreNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask));
      PetscCall(PetscFree2(func, ctx));
    }
    PetscCall(DMGetNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask));
  }
  ctx.dm   = dm;
  ctx.mask = mask;
  PetscCall(VecGetSize(y, &N));
  PetscCall(VecGetLocalSize(y, &n));
  PetscCall(MatCreate(PetscObjectComm((PetscObject)dm), &CtC));
  PetscCall(MatSetSizes(CtC, n, n, N, N));
  PetscCall(MatSetType(CtC, MATSHELL));
  PetscCall(MatSetUp(CtC));
  PetscCall(MatShellSetContext(CtC, &ctx));
  PetscCall(MatShellSetOperation(CtC, MATOP_MULT, (PetscErrorCodeFn *)MatMult_GlobalToLocalNormal));
  PetscCall(KSPCreate(PetscObjectComm((PetscObject)dm), &ksp));
  PetscCall(KSPSetOperators(ksp, CtC, CtC));
  PetscCall(KSPSetType(ksp, KSPCG));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(KSPSetUp(ksp));
  PetscCall(DMGetGlobalVector(dm, &global));
  PetscCall(VecSet(global, 0.));
  if (mask) PetscCall(VecPointwiseMult(x, mask, x));
  PetscCall(DMLocalToGlobalBegin(dm, x, ADD_VALUES, global));
  PetscCall(DMLocalToGlobalEnd(dm, x, ADD_VALUES, global));
  PetscCall(KSPSolve(ksp, global, y));
  PetscCall(DMRestoreGlobalVector(dm, &global));
  /* clean up */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&CtC));
  if (isPlex) {
    PetscCall(VecDestroy(&mask));
  } else {
    PetscCall(DMRestoreNamedLocalVector(dm, "_DMGlobalToLocalSolve_mask", &mask));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMProjectField - This projects a given function of the input fields into the function space provided by a `DM`, putting the coefficients in a global vector.

  Collective

  Input Parameters:
+ dm    - The `DM`
. time  - The time
. U     - The input field vector
. funcs - The functions to evaluate, one per field, see `PetscPointFn`
- mode  - The insertion mode for values

  Output Parameter:
. X - The output vector

  Level: advanced

  Note:
  There are three different `DM`s that potentially interact in this function. The output `dm`, specifies the layout of the values calculates by the function.
  The input `DM`, attached to `U`, may be different. For example, you can input the solution over the full domain, but output over a piece of the boundary, or
  a subdomain. You can also output a different number of fields than the input, with different discretizations. Last the auxiliary `DM`, attached to the
  auxiliary field vector, which is attached to `dm`, can also be different. It can have a different topology, number of fields, and discretizations.

.seealso: [](ch_dmbase), `DM`, `PetscPointFn`, `DMProjectFieldLocal()`, `DMProjectFieldLabelLocal()`, `DMProjectFunction()`, `DMComputeL2Diff()`
@*/
PetscErrorCode DMProjectField(DM dm, PetscReal time, Vec U, PetscPointFn **funcs, InsertMode mode, Vec X)
{
  Vec localX, localU;
  DM  dmIn;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  PetscCall(DMGetLocalVector(dm, &localX));
  /* We currently check whether locU == locX to see if we need to apply BC */
  if (U != X) {
    PetscCall(VecGetDM(U, &dmIn));
    PetscCall(DMGetLocalVector(dmIn, &localU));
  } else {
    dmIn   = dm;
    localU = localX;
  }
  PetscCall(DMGlobalToLocalBegin(dmIn, U, INSERT_VALUES, localU));
  PetscCall(DMGlobalToLocalEnd(dmIn, U, INSERT_VALUES, localU));
  PetscCall(DMProjectFieldLocal(dm, time, localU, funcs, mode, localX));
  PetscCall(DMLocalToGlobalBegin(dm, localX, mode, X));
  PetscCall(DMLocalToGlobalEnd(dm, localX, mode, X));
  if (mode == INSERT_VALUES || mode == INSERT_ALL_VALUES || mode == INSERT_BC_VALUES) {
    Mat cMat;

    PetscCall(DMGetDefaultConstraints(dm, NULL, &cMat, NULL));
    if (cMat) PetscCall(DMGlobalToLocalSolve(dm, localX, X));
  }
  PetscCall(DMRestoreLocalVector(dm, &localX));
  if (U != X) PetscCall(DMRestoreLocalVector(dmIn, &localU));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/********************* Adaptive Interpolation **************************/

/* See the discussion of Adaptive Interpolation in manual/high_level_mg.rst */
PetscErrorCode DMAdaptInterpolator(DM dmc, DM dmf, Mat In, KSP smoother, Mat MF, Mat MC, Mat *InAdapt, void *user)
{
  Mat                globalA, AF;
  Vec                tmp;
  const PetscScalar *af, *ac;
  PetscScalar       *A, *b, *x, *workscalar;
  PetscReal         *w, *sing, *workreal, rcond = PETSC_SMALL;
  PetscBLASInt       M, N, one = 1, irank, lwrk, info;
  PetscInt           debug = 0, rStart, rEnd, r, maxcols = 0, k, Nc, ldac, ldaf;
  PetscBool          allocVc = PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscLogEventBegin(DM_AdaptInterpolator, dmc, dmf, 0, 0));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dm_interpolator_adapt_debug", &debug, NULL));
  PetscCall(MatGetSize(MF, NULL, &Nc));
  PetscCall(MatDuplicate(In, MAT_SHARE_NONZERO_PATTERN, InAdapt));
  PetscCall(MatGetOwnershipRange(In, &rStart, &rEnd));
#if 0
  PetscCall(MatGetMaxRowLen(In, &maxcols));
#else
  for (r = rStart; r < rEnd; ++r) {
    PetscInt ncols;

    PetscCall(MatGetRow(In, r, &ncols, NULL, NULL));
    maxcols = PetscMax(maxcols, ncols);
    PetscCall(MatRestoreRow(In, r, &ncols, NULL, NULL));
  }
#endif
  if (Nc < maxcols) PetscCall(PetscPrintf(PETSC_COMM_SELF, "The number of input vectors %" PetscInt_FMT " < %" PetscInt_FMT " the maximum number of column entries\n", Nc, maxcols));
  for (k = 0; k < Nc && debug; ++k) {
    char        name[PETSC_MAX_PATH_LEN];
    const char *prefix;
    Vec         vc, vf;

    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)smoother, &prefix));

    if (MC) {
      PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "%sCoarse Vector %" PetscInt_FMT, prefix ? prefix : NULL, k));
      PetscCall(MatDenseGetColumnVecRead(MC, k, &vc));
      PetscCall(PetscObjectSetName((PetscObject)vc, name));
      PetscCall(VecViewFromOptions(vc, NULL, "-dm_adapt_interp_view_coarse"));
      PetscCall(MatDenseRestoreColumnVecRead(MC, k, &vc));
    }
    PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "%sFine Vector %" PetscInt_FMT, prefix ? prefix : NULL, k));
    PetscCall(MatDenseGetColumnVecRead(MF, k, &vf));
    PetscCall(PetscObjectSetName((PetscObject)vf, name));
    PetscCall(VecViewFromOptions(vf, NULL, "-dm_adapt_interp_view_fine"));
    PetscCall(MatDenseRestoreColumnVecRead(MF, k, &vf));
  }
  PetscCall(PetscBLASIntCast(3 * PetscMin(Nc, maxcols) + PetscMax(2 * PetscMin(Nc, maxcols), PetscMax(Nc, maxcols)), &lwrk));
  PetscCall(PetscMalloc7(Nc * maxcols, &A, PetscMax(Nc, maxcols), &b, Nc, &w, maxcols, &x, maxcols, &sing, lwrk, &workscalar, 5 * PetscMin(Nc, maxcols), &workreal));
  /* w_k = \frac{\HC{v_k} B_l v_k}{\HC{v_k} A_l v_k} or the inverse Rayleigh quotient, which we calculate using \frac{\HC{v_k} v_k}{\HC{v_k} B^{-1}_l A_l v_k} */
  PetscCall(KSPGetOperators(smoother, &globalA, NULL));

  PetscCall(MatMatMult(globalA, MF, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &AF));
  for (k = 0; k < Nc; ++k) {
    PetscScalar vnorm, vAnorm;
    Vec         vf;

    w[k] = 1.0;
    PetscCall(MatDenseGetColumnVecRead(MF, k, &vf));
    PetscCall(MatDenseGetColumnVecRead(AF, k, &tmp));
    PetscCall(VecDot(vf, vf, &vnorm));
#if 0
    PetscCall(DMGetGlobalVector(dmf, &tmp2));
    PetscCall(KSPSolve(smoother, tmp, tmp2));
    PetscCall(VecDot(vf, tmp2, &vAnorm));
    PetscCall(DMRestoreGlobalVector(dmf, &tmp2));
#else
    PetscCall(VecDot(vf, tmp, &vAnorm));
#endif
    w[k] = PetscRealPart(vnorm) / PetscRealPart(vAnorm);
    PetscCall(MatDenseRestoreColumnVecRead(MF, k, &vf));
    PetscCall(MatDenseRestoreColumnVecRead(AF, k, &tmp));
  }
  PetscCall(MatDestroy(&AF));
  if (!MC) {
    allocVc = PETSC_TRUE;
    PetscCall(MatTransposeMatMult(In, MF, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &MC));
  }
  /* Solve a LS system for each fine row
     MATT: Can we generalize to the case where Nc for the fine space
     is different for Nc for the coarse? */
  PetscCall(MatDenseGetArrayRead(MF, &af));
  PetscCall(MatDenseGetLDA(MF, &ldaf));
  PetscCall(MatDenseGetArrayRead(MC, &ac));
  PetscCall(MatDenseGetLDA(MC, &ldac));
  for (r = rStart; r < rEnd; ++r) {
    PetscInt           ncols, c;
    const PetscInt    *cols;
    const PetscScalar *vals;

    PetscCall(MatGetRow(In, r, &ncols, &cols, &vals));
    for (k = 0; k < Nc; ++k) {
      /* Need to fit lowest mode exactly */
      const PetscReal wk = ((ncols == 1) && (k > 0)) ? 0.0 : PetscSqrtReal(w[k]);

      /* b_k = \sqrt{w_k} f^{F,k}_r */
      b[k] = wk * af[r - rStart + k * ldaf];
      /* A_{kc} = \sqrt{w_k} f^{C,k}_c */
      /* TODO Must pull out VecScatter from In, scatter in vc[k] values up front, and access them indirectly just as in MatMult() */
      for (c = 0; c < ncols; ++c) {
        /* This is element (k, c) of A */
        A[c * Nc + k] = wk * ac[cols[c] - rStart + k * ldac];
      }
    }
    PetscCall(PetscBLASIntCast(Nc, &M));
    PetscCall(PetscBLASIntCast(ncols, &N));
    if (debug) {
#if defined(PETSC_USE_COMPLEX)
      PetscScalar *tmp;
      PetscInt     j;

      PetscCall(DMGetWorkArray(dmc, Nc, MPIU_SCALAR, (void *)&tmp));
      for (j = 0; j < Nc; ++j) tmp[j] = w[j];
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS weights", Nc, 1, tmp));
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS matrix", Nc, ncols, A));
      for (j = 0; j < Nc; ++j) tmp[j] = b[j];
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS rhs", Nc, 1, tmp));
      PetscCall(DMRestoreWorkArray(dmc, Nc, MPIU_SCALAR, (void *)&tmp));
#else
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS weights", Nc, 1, w));
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS matrix", Nc, ncols, A));
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS rhs", Nc, 1, b));
#endif
    }
#if defined(PETSC_USE_COMPLEX)
    /* ZGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, RWORK, INFO) */
    PetscCallBLAS("LAPACKgelss", LAPACKgelss_(&M, &N, &one, A, &M, b, M > N ? &M : &N, sing, &rcond, &irank, workscalar, &lwrk, workreal, &info));
#else
    /* DGELSS( M, N, NRHS, A, LDA, B, LDB, S, RCOND, RANK, WORK, LWORK, INFO) */
    PetscCallBLAS("LAPACKgelss", LAPACKgelss_(&M, &N, &one, A, &M, b, M > N ? &M : &N, sing, &rcond, &irank, workscalar, &lwrk, &info));
#endif
    PetscCheck(info >= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "Bad argument to GELSS");
    PetscCheck(info <= 0, PETSC_COMM_SELF, PETSC_ERR_LIB, "SVD failed to converge");
    if (debug) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "rank %" PetscBLASInt_FMT " rcond %g\n", irank, (double)rcond));
#if defined(PETSC_USE_COMPLEX)
      {
        PetscScalar *tmp;
        PetscInt     j;

        PetscCall(DMGetWorkArray(dmc, Nc, MPIU_SCALAR, (void *)&tmp));
        for (j = 0; j < PetscMin(Nc, ncols); ++j) tmp[j] = sing[j];
        PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS singular values", PetscMin(Nc, ncols), 1, tmp));
        PetscCall(DMRestoreWorkArray(dmc, Nc, MPIU_SCALAR, (void *)&tmp));
      }
#else
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS singular values", PetscMin(Nc, ncols), 1, sing));
#endif
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS old P", ncols, 1, vals));
      PetscCall(DMPrintCellMatrix(r, "Interpolator Row LS sol", ncols, 1, b));
    }
    PetscCall(MatSetValues(*InAdapt, 1, &r, ncols, cols, b, INSERT_VALUES));
    PetscCall(MatRestoreRow(In, r, &ncols, &cols, &vals));
  }
  PetscCall(MatDenseRestoreArrayRead(MF, &af));
  PetscCall(MatDenseRestoreArrayRead(MC, &ac));
  PetscCall(PetscFree7(A, b, w, x, sing, workscalar, workreal));
  if (allocVc) PetscCall(MatDestroy(&MC));
  PetscCall(MatAssemblyBegin(*InAdapt, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(*InAdapt, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscLogEventEnd(DM_AdaptInterpolator, dmc, dmf, 0, 0));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMCheckInterpolator(DM dmf, Mat In, Mat MC, Mat MF, PetscReal tol)
{
  Vec       tmp;
  PetscReal norminf, norm2, maxnorminf = 0.0, maxnorm2 = 0.0;
  PetscInt  k, Nc;

  PetscFunctionBegin;
  PetscCall(DMGetGlobalVector(dmf, &tmp));
  PetscCall(MatViewFromOptions(In, NULL, "-dm_interpolator_adapt_error"));
  PetscCall(MatGetSize(MF, NULL, &Nc));
  for (k = 0; k < Nc; ++k) {
    Vec vc, vf;

    PetscCall(MatDenseGetColumnVecRead(MC, k, &vc));
    PetscCall(MatDenseGetColumnVecRead(MF, k, &vf));
    PetscCall(MatMult(In, vc, tmp));
    PetscCall(VecAXPY(tmp, -1.0, vf));
    PetscCall(VecViewFromOptions(vc, NULL, "-dm_interpolator_adapt_error"));
    PetscCall(VecViewFromOptions(vf, NULL, "-dm_interpolator_adapt_error"));
    PetscCall(VecViewFromOptions(tmp, NULL, "-dm_interpolator_adapt_error"));
    PetscCall(VecNorm(tmp, NORM_INFINITY, &norminf));
    PetscCall(VecNorm(tmp, NORM_2, &norm2));
    maxnorminf = PetscMax(maxnorminf, norminf);
    maxnorm2   = PetscMax(maxnorm2, norm2);
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dmf), "Coarse vec %" PetscInt_FMT " ||vf - P vc||_\\infty %g, ||vf - P vc||_2 %g\n", k, (double)norminf, (double)norm2));
    PetscCall(MatDenseRestoreColumnVecRead(MC, k, &vc));
    PetscCall(MatDenseRestoreColumnVecRead(MF, k, &vf));
  }
  PetscCall(DMRestoreGlobalVector(dmf, &tmp));
  PetscCheck(maxnorm2 <= tol, PetscObjectComm((PetscObject)dmf), PETSC_ERR_ARG_WRONG, "max_k ||vf_k - P vc_k||_2 %g > tol %g", (double)maxnorm2, (double)tol);
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Project particles to field
//   M_f u_f = M_p u_p
//   u_f = M^{-1}_f M_p u_p
static PetscErrorCode DMSwarmProjectField_Conservative_PLEX(DM sw, DM dm, Vec u_p, Vec u_f)
{
  KSP         ksp;
  Mat         M_f, M_p; // TODO Should cache these
  Vec         rhs;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(DMCreateMassMatrix(dm, dm, &M_f));
  PetscCall(DMCreateMassMatrix(sw, dm, &M_p));
  PetscCall(DMGetGlobalVector(dm, &rhs));
  PetscCall(MatMultTranspose(M_p, u_p, rhs));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)sw), &ksp));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)sw, &prefix));
  PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(ksp, "ptof_"));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSetOperators(ksp, M_f, M_f));
  PetscCall(KSPSolve(ksp, rhs, u_f));

  PetscCall(DMRestoreGlobalVector(dm, &rhs));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&M_f));
  PetscCall(MatDestroy(&M_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Project field to particles
//   M_p u_p = M_f u_f
//   u_p = M^+_p M_f u_f
static PetscErrorCode DMSwarmProjectParticles_Conservative_PLEX(DM sw, DM dm, Vec u_p, Vec u_f)
{
  KSP         ksp;
  PC          pc;
  Mat         M_f, M_p, PM_p;
  Vec         rhs;
  PetscBool   isBjacobi;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(DMCreateMassMatrix(dm, dm, &M_f));
  PetscCall(DMCreateMassMatrix(sw, dm, &M_p));
  PetscCall(DMGetGlobalVector(dm, &rhs));
  PetscCall(MatMult(M_f, u_f, rhs));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)sw), &ksp));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)sw, &prefix));
  PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(ksp, "ftop_"));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCBJACOBI, &isBjacobi));
  if (isBjacobi) {
    PetscCall(DMSwarmCreateMassMatrixSquare(sw, dm, &PM_p));
  } else {
    PM_p = M_p;
    PetscCall(PetscObjectReference((PetscObject)PM_p));
  }
  PetscCall(KSPSetOperators(ksp, M_p, PM_p));
  PetscCall(KSPSolveTranspose(ksp, rhs, u_p));

  PetscCall(DMRestoreGlobalVector(dm, &rhs));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&M_f));
  PetscCall(MatDestroy(&M_p));
  PetscCall(MatDestroy(&PM_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmProjectFields_Plex_Internal(DM sw, DM dm, PetscInt Nf, const char *fieldnames[], Vec vec, ScatterMode mode)
{
  PetscDS  ds;
  Vec      u;
  PetscInt f = 0, bs, *Nc;

  PetscFunctionBegin;
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetComponents(ds, &Nc));
  PetscCall(PetscCitationsRegister(SwarmProjCitation, &SwarmProjcite));
  PetscCheck(Nf == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "Currently supported only for a single field");
  PetscCall(DMSwarmVectorDefineFields(sw, Nf, fieldnames));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, fieldnames[f], &u));
  PetscCall(VecGetBlockSize(u, &bs));
  PetscCheck(Nc[f] == bs, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "Field %" PetscInt_FMT " components %" PetscInt_FMT " != %" PetscInt_FMT " blocksize for swarm field %s", f, Nc[f], bs, fieldnames[f]);
  if (mode == SCATTER_FORWARD) {
    PetscCall(DMSwarmProjectField_Conservative_PLEX(sw, dm, u, vec));
  } else {
    PetscCall(DMSwarmProjectParticles_Conservative_PLEX(sw, dm, u, vec));
  }
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, fieldnames[0], &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmProjectField_ApproxQ1_DA_2D(DM swarm, PetscReal *swarm_field, DM dm, Vec v_field)
{
  DMSwarmCellDM      celldm;
  Vec                v_field_l, denom_l, coor_l, denom;
  PetscScalar       *_field_l, *_denom_l;
  PetscInt           k, p, e, npoints, nel, npe, Nfc;
  PetscInt          *mpfield_cell;
  PetscReal         *mpfield_coor;
  const PetscInt    *element_list;
  const PetscInt    *element;
  PetscScalar        xi_p[2], Ni[4];
  const PetscScalar *_coor;
  const char       **coordFields, *cellid;

  PetscFunctionBegin;
  PetscCall(VecZeroEntries(v_field));

  PetscCall(DMGetLocalVector(dm, &v_field_l));
  PetscCall(DMGetGlobalVector(dm, &denom));
  PetscCall(DMGetLocalVector(dm, &denom_l));
  PetscCall(VecZeroEntries(v_field_l));
  PetscCall(VecZeroEntries(denom));
  PetscCall(VecZeroEntries(denom_l));

  PetscCall(VecGetArray(v_field_l, &_field_l));
  PetscCall(VecGetArray(denom_l, &_denom_l));

  PetscCall(DMGetCoordinatesLocal(dm, &coor_l));
  PetscCall(VecGetArrayRead(coor_l, &_coor));

  PetscCall(DMSwarmGetCellDMActive(swarm, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PetscObjectComm((PetscObject)swarm), PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);
  PetscCall(DMSwarmCellDMGetCellID(celldm, &cellid));

  PetscCall(DMDAGetElements(dm, &nel, &npe, &element_list));
  PetscCall(DMSwarmGetLocalSize(swarm, &npoints));
  PetscCall(DMSwarmGetField(swarm, coordFields[0], NULL, NULL, (void **)&mpfield_coor));
  PetscCall(DMSwarmGetField(swarm, cellid, NULL, NULL, (void **)&mpfield_cell));

  for (p = 0; p < npoints; p++) {
    PetscReal         *coor_p;
    const PetscScalar *x0;
    const PetscScalar *x2;
    PetscScalar        dx[2];

    e       = mpfield_cell[p];
    coor_p  = &mpfield_coor[2 * p];
    element = &element_list[npe * e];

    /* compute local coordinates: (xp-x0)/dx = (xip+1)/2 */
    x0 = &_coor[2 * element[0]];
    x2 = &_coor[2 * element[2]];

    dx[0] = x2[0] - x0[0];
    dx[1] = x2[1] - x0[1];

    xi_p[0] = 2.0 * (coor_p[0] - x0[0]) / dx[0] - 1.0;
    xi_p[1] = 2.0 * (coor_p[1] - x0[1]) / dx[1] - 1.0;

    /* evaluate basis functions */
    Ni[0] = 0.25 * (1.0 - xi_p[0]) * (1.0 - xi_p[1]);
    Ni[1] = 0.25 * (1.0 + xi_p[0]) * (1.0 - xi_p[1]);
    Ni[2] = 0.25 * (1.0 + xi_p[0]) * (1.0 + xi_p[1]);
    Ni[3] = 0.25 * (1.0 - xi_p[0]) * (1.0 + xi_p[1]);

    for (k = 0; k < npe; k++) {
      _field_l[element[k]] += Ni[k] * swarm_field[p];
      _denom_l[element[k]] += Ni[k];
    }
  }

  PetscCall(DMSwarmRestoreField(swarm, cellid, NULL, NULL, (void **)&mpfield_cell));
  PetscCall(DMSwarmRestoreField(swarm, coordFields[0], NULL, NULL, (void **)&mpfield_coor));
  PetscCall(DMDARestoreElements(dm, &nel, &npe, &element_list));
  PetscCall(VecRestoreArrayRead(coor_l, &_coor));
  PetscCall(VecRestoreArray(v_field_l, &_field_l));
  PetscCall(VecRestoreArray(denom_l, &_denom_l));

  PetscCall(DMLocalToGlobalBegin(dm, v_field_l, ADD_VALUES, v_field));
  PetscCall(DMLocalToGlobalEnd(dm, v_field_l, ADD_VALUES, v_field));
  PetscCall(DMLocalToGlobalBegin(dm, denom_l, ADD_VALUES, denom));
  PetscCall(DMLocalToGlobalEnd(dm, denom_l, ADD_VALUES, denom));

  PetscCall(VecPointwiseDivide(v_field, v_field, denom));

  PetscCall(DMRestoreLocalVector(dm, &v_field_l));
  PetscCall(DMRestoreLocalVector(dm, &denom_l));
  PetscCall(DMRestoreGlobalVector(dm, &denom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmProjectFields_DA_Internal(DM swarm, DM celldm, PetscInt nfields, DMSwarmDataField dfield[], Vec vecs[], ScatterMode mode)
{
  PetscInt        f, dim;
  DMDAElementType etype;

  PetscFunctionBegin;
  PetscCall(DMDAGetElementType(celldm, &etype));
  PetscCheck(etype != DMDA_ELEMENT_P1, PetscObjectComm((PetscObject)swarm), PETSC_ERR_SUP, "Only Q1 DMDA supported");
  PetscCheck(mode == SCATTER_FORWARD, PetscObjectComm((PetscObject)swarm), PETSC_ERR_SUP, "Mapping the continuum to particles is not currently supported for DMDA");

  PetscCall(DMGetDimension(swarm, &dim));
  switch (dim) {
  case 2:
    for (f = 0; f < nfields; f++) {
      PetscReal *swarm_field;

      PetscCall(DMSwarmDataFieldGetEntries(dfield[f], (void **)&swarm_field));
      PetscCall(DMSwarmProjectField_ApproxQ1_DA_2D(swarm, swarm_field, celldm, vecs[f]));
    }
    break;
  case 3:
    SETERRQ(PetscObjectComm((PetscObject)swarm), PETSC_ERR_SUP, "No support for 3D");
  default:
    break;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMSwarmProjectFields - Project a set of swarm fields onto another `DM`

  Collective

  Input Parameters:
+ sw         - the `DMSWARM`
. dm         - the `DM`, or `NULL` to use the cell `DM`
. nfields    - the number of swarm fields to project
. fieldnames - the textual names of the swarm fields to project
. fields     - an array of `Vec`'s of length nfields
- mode       - if `SCATTER_FORWARD` then map particles to the continuum, and if `SCATTER_REVERSE` map the continuum to particles

  Level: beginner

  Notes:
  Currently, there are two available projection methods. The first is conservative projection, used for a `DMPLEX` cell `DM`.
  The second is the averaging which is used for a `DMDA` cell `DM`

  $$
  \phi_i = \sum_{p=0}^{np} N_i(x_p) \phi_p dJ / \sum_{p=0}^{np} N_i(x_p) dJ
  $$

  where $\phi_p $ is the swarm field at point $p$, $N_i()$ is the cell `DM` basis function at vertex $i$, $dJ$ is the determinant of the cell Jacobian and
  $\phi_i$ is the projected vertex value of the field $\phi$.

  The user is responsible for destroying both the array and the individual `Vec` objects.

  For the `DMPLEX` case, there is only a single vector, so the field layout in the `DMPLEX` must match the requested fields from the `DMSwarm`.

  For averaging projection, nly swarm fields registered with data type of `PETSC_REAL` can be projected onto the cell `DM`, and only swarm fields of block size = 1 can currently be projected.

.seealso: [](ch_dmbase), `DMSWARM`, `DMSwarmSetType()`, `DMSwarmSetCellDM()`, `DMSwarmType`
@*/
PetscErrorCode DMSwarmProjectFields(DM sw, DM dm, PetscInt nfields, const char *fieldnames[], Vec fields[], ScatterMode mode)
{
  DM_Swarm         *swarm = (DM_Swarm *)sw->data;
  DMSwarmDataField *gfield;
  PetscBool         isDA, isPlex;
  MPI_Comm          comm;

  PetscFunctionBegin;
  DMSWARMPICVALID(sw);
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));
  if (!dm) PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMDA, &isDA));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isPlex));
  PetscCall(PetscMalloc1(nfields, &gfield));
  for (PetscInt f = 0; f < nfields; ++f) PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, fieldnames[f], &gfield[f]));

  if (isDA) {
    for (PetscInt f = 0; f < nfields; f++) {
      PetscCheck(gfield[f]->petsc_type == PETSC_REAL, comm, PETSC_ERR_SUP, "Projection only valid for fields using a data type = PETSC_REAL");
      PetscCheck(gfield[f]->bs == 1, comm, PETSC_ERR_SUP, "Projection only valid for fields with block size = 1");
    }
    PetscCall(DMSwarmProjectFields_DA_Internal(sw, dm, nfields, gfield, fields, mode));
  } else if (isPlex) {
    PetscInt Nf;

    PetscCall(DMGetNumFields(dm, &Nf));
    PetscCheck(Nf == nfields, comm, PETSC_ERR_ARG_WRONG, "Number of DM fields %" PetscInt_FMT " != %" PetscInt_FMT " number of requested Swarm fields", Nf, nfields);
    PetscCall(DMSwarmProjectFields_Plex_Internal(sw, dm, nfields, fieldnames, fields[0], mode));
  } else SETERRQ(PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "Only supported for cell DMs of type DMDA and DMPLEX");

  PetscCall(PetscFree(gfield));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Project weak divergence of particles to field
//   \int_X psi_i div u_f = \int_X psi_i div u_p
//   \int_X grad psi_i . \sum_j u_f \psi_j = \int_X grad psi_i . \sum_p u_p \delta(x - x_p)
//   D_f u_f = D_p u_p
//   u_f = D^+_f D_p u_p
static PetscErrorCode DMSwarmProjectGradientField_Conservative_PLEX(DM sw, DM dm, Vec u_p, Vec u_f)
{
  DM          gdm;
  KSP         ksp;
  Mat         D_f, D_p; // TODO Should cache these
  Vec         rhs;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(VecGetDM(u_f, &gdm));
  PetscCall(DMCreateGradientMatrix(dm, gdm, &D_f));
  PetscCall(DMCreateGradientMatrix(sw, dm, &D_p));
  PetscCall(DMGetGlobalVector(dm, &rhs));
  PetscCall(PetscObjectSetName((PetscObject)rhs, "D u"));
  PetscCall(MatMultTranspose(D_p, u_p, rhs));
  PetscCall(VecViewFromOptions(rhs, NULL, "-rhs_view"));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)sw), &ksp));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)sw, &prefix));
  PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(ksp, "gptof_"));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPSetOperators(ksp, D_f, D_f));
  PetscCall(KSPSolveTranspose(ksp, rhs, u_f));

  PetscCall(MatMultTranspose(D_f, u_f, rhs));
  PetscCall(VecViewFromOptions(rhs, NULL, "-rhs_view"));

  PetscCall(DMRestoreGlobalVector(dm, &rhs));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&D_f));
  PetscCall(MatDestroy(&D_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Project weak divergence of field to particles
//   D_p u_p = D_f u_f
//   u_p = D^+_p D_f u_f
static PetscErrorCode DMSwarmProjectGradientParticles_Conservative_PLEX(DM sw, DM dm, Vec u_p, Vec u_f)
{
  KSP         ksp;
  PC          pc;
  Mat         D_f, D_p, PD_p;
  Vec         rhs;
  PetscBool   isBjacobi;
  const char *prefix;

  PetscFunctionBegin;
  PetscCall(DMCreateGradientMatrix(dm, dm, &D_f));
  PetscCall(DMCreateGradientMatrix(sw, dm, &D_p));
  PetscCall(DMGetGlobalVector(dm, &rhs));
  PetscCall(MatMult(D_f, u_f, rhs));

  PetscCall(KSPCreate(PetscObjectComm((PetscObject)sw), &ksp));
  PetscCall(PetscObjectGetOptionsPrefix((PetscObject)sw, &prefix));
  PetscCall(KSPSetOptionsPrefix(ksp, prefix));
  PetscCall(KSPAppendOptionsPrefix(ksp, "gftop_"));
  PetscCall(KSPSetFromOptions(ksp));

  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCBJACOBI, &isBjacobi));
  if (isBjacobi) {
    PetscCall(DMSwarmCreateMassMatrixSquare(sw, dm, &PD_p));
  } else {
    PD_p = D_p;
    PetscCall(PetscObjectReference((PetscObject)PD_p));
  }
  PetscCall(KSPSetOperators(ksp, D_p, PD_p));
  PetscCall(KSPSolveTranspose(ksp, rhs, u_p));

  PetscCall(DMRestoreGlobalVector(dm, &rhs));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&D_f));
  PetscCall(MatDestroy(&D_p));
  PetscCall(MatDestroy(&PD_p));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmProjectGradientFields_Plex_Internal(DM sw, DM dm, PetscInt Nf, const char *fieldnames[], Vec vec, ScatterMode mode)
{
  PetscDS  ds;
  Vec      u;
  PetscInt f = 0, cdim, bs, *Nc;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetDS(dm, &ds));
  PetscCall(PetscDSGetComponents(ds, &Nc));
  PetscCall(PetscCitationsRegister(SwarmProjCitation, &SwarmProjcite));
  PetscCheck(Nf == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "Currently supported only for a single field");
  PetscCall(DMSwarmVectorDefineFields(sw, Nf, fieldnames));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, fieldnames[f], &u));
  PetscCall(VecGetBlockSize(u, &bs));
  PetscCheck(Nc[f] * cdim == bs, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "Field %" PetscInt_FMT " components %" PetscInt_FMT " * %" PetscInt_FMT " coordinate dim != %" PetscInt_FMT " blocksize for swarm field %s", f, Nc[f], cdim, bs, fieldnames[f]);
  if (mode == SCATTER_FORWARD) {
    PetscCall(DMSwarmProjectGradientField_Conservative_PLEX(sw, dm, u, vec));
  } else {
    PetscCall(DMSwarmProjectGradientParticles_Conservative_PLEX(sw, dm, u, vec));
  }
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, fieldnames[0], &u));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode DMSwarmProjectGradientFields(DM sw, DM dm, PetscInt nfields, const char *fieldnames[], Vec fields[], ScatterMode mode)
{
  PetscBool isPlex;
  MPI_Comm  comm;

  PetscFunctionBegin;
  DMSWARMPICVALID(sw);
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));
  if (!dm) PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(PetscObjectTypeCompare((PetscObject)dm, DMPLEX, &isPlex));
  if (isPlex) {
    PetscInt Nf;

    PetscCall(DMGetNumFields(dm, &Nf));
    PetscCheck(Nf == nfields, comm, PETSC_ERR_ARG_WRONG, "Number of DM fields %" PetscInt_FMT " != %" PetscInt_FMT " number of requested Swarm fields", Nf, nfields);
    PetscCall(DMSwarmProjectGradientFields_Plex_Internal(sw, dm, nfields, fieldnames, fields[0], mode));
  } else SETERRQ(PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "Only supported for cell DMs of type DMPLEX");
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  InitializeParticles_Regular - Initialize a regular grid of particles in each cell

  Input Parameters:
+ sw - The `DMSWARM`
- n  - The number of particles per dimension per species

Notes:
  This functions sets the species, cellid, and cell DM coordinates.

  It places n^d particles per species in each cell of the cell DM.
*/
static PetscErrorCode InitializeParticles_Regular(DM sw, PetscInt n)
{
  DM_Swarm     *swarm = (DM_Swarm *)sw->data;
  DM            dm;
  DMSwarmCellDM celldm;
  PetscInt      dim, Ns, Npc, Np, cStart, cEnd, debug;
  PetscBool     flg;
  MPI_Comm      comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)sw, &comm));

  PetscOptionsBegin(comm, "", "DMSwarm Options", "DMSWARM");
  PetscCall(DMSwarmGetNumSpecies(sw, &Ns));
  PetscCall(PetscOptionsInt("-dm_swarm_num_species", "The number of species", "DMSwarmSetNumSpecies", Ns, &Ns, &flg));
  if (flg) PetscCall(DMSwarmSetNumSpecies(sw, Ns));
  PetscCall(PetscOptionsBoundedInt("-dm_swarm_print_coords", "Debug output level for particle coordinate computations", "InitializeParticles", 0, &swarm->printCoords, NULL, 0));
  PetscCall(PetscOptionsBoundedInt("-dm_swarm_print_weights", "Debug output level for particle weight computations", "InitializeWeights", 0, &swarm->printWeights, NULL, 0));
  PetscOptionsEnd();
  debug = swarm->printCoords;

  // n^d particle per cell on the grid
  PetscCall(DMSwarmGetCellDM(sw, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCheck(!(dim % 2), comm, PETSC_ERR_SUP, "We only support even dimension, not %" PetscInt_FMT, dim);
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  Npc = Ns * PetscPowInt(n, dim);
  Np  = (cEnd - cStart) * Npc;
  PetscCall(DMSwarmSetLocalSizes(sw, Np, 0));
  if (debug) {
    PetscInt gNp;
    PetscCallMPI(MPIU_Allreduce(&Np, &gNp, 1, MPIU_INT, MPIU_SUM, comm));
    PetscCall(PetscPrintf(comm, "Global Np = %" PetscInt_FMT "\n", gNp));
  }
  PetscCall(PetscPrintf(comm, "Regular layout using %" PetscInt_FMT " particles per cell\n", Npc));

  // Set species and cellid
  {
    const char *cellidName;
    PetscInt   *species, *cellid;

    PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
    PetscCall(DMSwarmCellDMGetCellID(celldm, &cellidName));
    PetscCall(DMSwarmGetField(sw, "species", NULL, NULL, (void **)&species));
    PetscCall(DMSwarmGetField(sw, cellidName, NULL, NULL, (void **)&cellid));
    for (PetscInt c = 0, p = 0; c < cEnd - cStart; ++c) {
      for (PetscInt s = 0; s < Ns; ++s) {
        for (PetscInt q = 0; q < Npc / Ns; ++q, ++p) {
          species[p] = s;
          cellid[p]  = c;
        }
      }
    }
    PetscCall(DMSwarmRestoreField(sw, "species", NULL, NULL, (void **)&species));
    PetscCall(DMSwarmRestoreField(sw, cellidName, NULL, NULL, (void **)&cellid));
  }

  // Set particle coordinates
  {
    PetscReal     *x, *v;
    const char   **coordNames;
    PetscInt       Ncoord;
    const PetscInt xdim = dim / 2, vdim = dim / 2;

    PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Ncoord, &coordNames));
    PetscCheck(Ncoord == 2, comm, PETSC_ERR_SUP, "We only support regular layout for 2 coordinate fields, not %" PetscInt_FMT, Ncoord);
    PetscCall(DMSwarmGetField(sw, coordNames[0], NULL, NULL, (void **)&x));
    PetscCall(DMSwarmGetField(sw, coordNames[1], NULL, NULL, (void **)&v));
    PetscCall(DMSwarmSortGetAccess(sw));
    PetscCall(DMGetCoordinatesLocalSetUp(dm));
    for (PetscInt c = 0; c < cEnd - cStart; ++c) {
      const PetscInt     cell = c + cStart;
      const PetscScalar *a;
      PetscScalar       *coords;
      PetscReal          lower[6], upper[6];
      PetscBool          isDG;
      PetscInt          *pidx, npc, Nc;

      PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &npc, &pidx));
      PetscCheck(Npc == npc, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Invalid number of points per cell %" PetscInt_FMT " != %" PetscInt_FMT, npc, Npc);
      PetscCall(DMPlexGetCellCoordinates(dm, cell, &isDG, &Nc, &a, &coords));
      for (PetscInt d = 0; d < dim; ++d) {
        lower[d] = PetscRealPart(coords[0 * dim + d]);
        upper[d] = PetscRealPart(coords[0 * dim + d]);
      }
      for (PetscInt i = 1; i < Nc / dim; ++i) {
        for (PetscInt d = 0; d < dim; ++d) {
          lower[d] = PetscMin(lower[d], PetscRealPart(coords[i * dim + d]));
          upper[d] = PetscMax(upper[d], PetscRealPart(coords[i * dim + d]));
        }
      }
      for (PetscInt s = 0; s < Ns; ++s) {
        for (PetscInt q = 0; q < Npc / Ns; ++q) {
          const PetscInt p = pidx[q * Ns + s];
          PetscInt       xi[3], vi[3];

          xi[0] = q % n;
          xi[1] = (q / n) % n;
          xi[2] = (q / PetscSqr(n)) % n;
          for (PetscInt d = 0; d < xdim; ++d) x[p * xdim + d] = lower[d] + (xi[d] + 0.5) * (upper[d] - lower[d]) / n;
          vi[0] = (q / PetscPowInt(n, xdim)) % n;
          vi[1] = (q / PetscPowInt(n, xdim + 1)) % n;
          vi[2] = (q / PetscPowInt(n, xdim + 2));
          for (PetscInt d = 0; d < vdim; ++d) v[p * vdim + d] = lower[xdim + d] + (vi[d] + 0.5) * (upper[xdim + d] - lower[xdim + d]) / n;
          if (debug > 1) {
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "Particle %4" PetscInt_FMT " ", p));
            PetscCall(PetscPrintf(PETSC_COMM_SELF, "  x: ("));
            for (PetscInt d = 0; d < xdim; ++d) {
              if (d > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
              PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", (double)PetscRealPart(x[p * xdim + d])));
            }
            PetscCall(PetscPrintf(PETSC_COMM_SELF, ") v:("));
            for (PetscInt d = 0; d < vdim; ++d) {
              if (d > 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, ", "));
              PetscCall(PetscPrintf(PETSC_COMM_SELF, "%g", (double)PetscRealPart(v[p * vdim + d])));
            }
            PetscCall(PetscPrintf(PETSC_COMM_SELF, ")\n"));
          }
        }
      }
      PetscCall(DMPlexRestoreCellCoordinates(dm, cell, &isDG, &Nc, &a, &coords));
      PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
    }
    PetscCall(DMSwarmSortRestoreAccess(sw));
    PetscCall(DMSwarmRestoreField(sw, coordNames[0], NULL, NULL, (void **)&x));
    PetscCall(DMSwarmRestoreField(sw, coordNames[1], NULL, NULL, (void **)&v));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
@article{MyersColellaVanStraalen2017,
   title   = {A 4th-order particle-in-cell method with phase-space remapping for the {Vlasov-Poisson} equation},
   author  = {Andrew Myers and Phillip Colella and Brian Van Straalen},
   journal = {SIAM Journal on Scientific Computing},
   volume  = {39},
   issue   = {3},
   pages   = {B467-B485},
   doi     = {10.1137/16M105962X},
   issn    = {10957197},
   year    = {2017},
}
*/
static PetscErrorCode W_3_Interpolation_Private(PetscReal x, PetscReal *w)
{
  const PetscReal ax = PetscAbsReal(x);

  PetscFunctionBegin;
  *w = 0.;
  // W_3(x) = 1 - 5/2 |x|^2 + 3/2 |x|^3   0 \le |x| \e 1
  if (ax <= 1.) *w = 1. - 2.5 * PetscSqr(ax) + 1.5 * PetscSqr(ax) * ax;
  //          1/2 (2 - |x|)^2 (1 - |x|)   1 \le |x| \le 2
  else if (ax <= 2.) *w = 0.5 * PetscSqr(2. - ax) * (1. - ax);
  //PetscCall(PetscPrintf(PETSC_COMM_SELF, "    W_3 %g --> %g\n", x, *w));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Right now, we will assume that the spatial and velocity grids are regular, which will speed up point location immensely
static PetscErrorCode DMSwarmRemap_Colella_Internal(DM sw, DM *rsw)
{
  DM            xdm, vdm;
  DMSwarmCellDM celldm;
  PetscReal     xmin[3], xmax[3], vmin[3], vmax[3];
  PetscInt      xend[3], vend[3];
  PetscReal    *x, *v, *w, *rw;
  PetscReal     hx[3], hv[3];
  PetscInt      dim, xcdim, vcdim, xcStart, xcEnd, vcStart, vcEnd, Np, Nfc;
  PetscInt      debug = ((DM_Swarm *)sw->data)->printWeights;
  const char  **coordFields;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(sw, &dim));
  PetscCall(DMSwarmGetCellDM(sw, &xdm));
  PetscCall(DMGetCoordinateDim(xdm, &xcdim));
  // Create a new centroid swarm without weights
  PetscCall(DMSwarmDuplicate(sw, rsw));
  PetscCall(DMSwarmGetCellDMActive(*rsw, &celldm));
  PetscCall(DMSwarmSetCellDMActive(*rsw, "remap"));
  PetscCall(InitializeParticles_Regular(*rsw, 1));
  PetscCall(DMSwarmSetCellDMActive(*rsw, ((PetscObject)celldm)->name));
  PetscCall(DMSwarmGetLocalSize(*rsw, &Np));
  // Assume quad mesh and calculate cell diameters (TODO this could be more robust)
  {
    const PetscScalar *array;
    PetscScalar       *coords;
    PetscBool          isDG;
    PetscInt           Nc;

    PetscCall(DMGetBoundingBox(xdm, xmin, xmax));
    PetscCall(DMPlexGetHeightStratum(xdm, 0, &xcStart, &xcEnd));
    PetscCall(DMPlexGetCellCoordinates(xdm, xcStart, &isDG, &Nc, &array, &coords));
    hx[0] = PetscRealPart(coords[1 * xcdim + 0] - coords[0 * xcdim + 0]);
    hx[1] = xcdim > 1 ? PetscRealPart(coords[2 * xcdim + 1] - coords[1 * xcdim + 1]) : 1.;
    PetscCall(DMPlexRestoreCellCoordinates(xdm, xcStart, &isDG, &Nc, &array, &coords));
    PetscCall(PetscObjectQuery((PetscObject)sw, "__vdm__", (PetscObject *)&vdm));
    PetscCall(DMGetCoordinateDim(vdm, &vcdim));
    PetscCall(DMGetBoundingBox(vdm, vmin, vmax));
    PetscCall(DMPlexGetHeightStratum(vdm, 0, &vcStart, &vcEnd));
    PetscCall(DMPlexGetCellCoordinates(vdm, vcStart, &isDG, &Nc, &array, &coords));
    hv[0] = PetscRealPart(coords[1 * vcdim + 0] - coords[0 * vcdim + 0]);
    hv[1] = vcdim > 1 ? PetscRealPart(coords[2 * vcdim + 1] - coords[1 * vcdim + 1]) : 1.;
    PetscCall(DMPlexRestoreCellCoordinates(vdm, vcStart, &isDG, &Nc, &array, &coords));

    PetscCheck(dim == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_ARG_WRONG, "Only support 1D distributions at this time");
    xend[0] = xcEnd - xcStart;
    xend[1] = 1;
    vend[0] = vcEnd - vcStart;
    vend[1] = 1;
    if (debug > 1)
      PetscCall(PetscPrintf(PETSC_COMM_SELF, "Phase Grid (%g, %g, %g, %g) (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n", (double)PetscRealPart(hx[0]), (double)PetscRealPart(hx[1]), (double)PetscRealPart(hv[0]), (double)PetscRealPart(hv[1]), xend[0], xend[1], vend[0], vend[1]));
  }
  // Iterate over particles in the original swarm
  PetscCall(DMSwarmGetCellDMActive(sw, &celldm));
  PetscCall(DMSwarmCellDMGetCoordinateFields(celldm, &Nfc, &coordFields));
  PetscCheck(Nfc == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "We only support a single coordinate field right now, not %" PetscInt_FMT, Nfc);
  PetscCall(DMSwarmGetField(sw, coordFields[0], NULL, NULL, (void **)&x));
  PetscCall(DMSwarmGetField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmGetField(sw, "w_q", NULL, NULL, (void **)&w));
  PetscCall(DMSwarmGetField(*rsw, "w_q", NULL, NULL, (void **)&rw));
  PetscCall(DMSwarmSortGetAccess(sw));
  PetscCall(DMSwarmSortGetAccess(*rsw));
  PetscCall(DMGetBoundingBox(vdm, vmin, vmax));
  PetscCall(DMGetCoordinatesLocalSetUp(xdm));
  for (PetscInt i = 0; i < Np; ++i) rw[i] = 0.;
  for (PetscInt c = 0; c < xcEnd - xcStart; ++c) {
    PetscInt *pidx, Npc;
    PetscInt *rpidx, rNpc;

    PetscCall(DMSwarmSortGetPointsPerCell(sw, c, &Npc, &pidx));
    for (PetscInt q = 0; q < Npc; ++q) {
      const PetscInt  p  = pidx[q];
      const PetscReal wp = w[p];
      PetscReal       Wx[3], Wv[3];
      PetscInt        xs[3], vs[3];

      // Determine the containing cell
      for (PetscInt d = 0; d < dim; ++d) {
        const PetscReal xp = x[p * dim + d];
        const PetscReal vp = v[p * dim + d];

        xs[d] = PetscFloorReal((xp - xmin[d]) / hx[d]);
        vs[d] = PetscFloorReal((vp - vmin[d]) / hv[d]);
      }
      // Loop over all grid points within 2 spacings of the particle
      if (debug > 2) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "Interpolating particle %" PetscInt_FMT " wt %g (%g, %g, %g, %g) (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ")\n", p, (double)wp, (double)PetscRealPart(x[p * dim + 0]), xcdim > 1 ? (double)PetscRealPart(x[p * xcdim + 1]) : 0., (double)PetscRealPart(v[p * vcdim + 0]), vcdim > 1 ? (double)PetscRealPart(v[p * vcdim + 1]) : 0., xs[0], xs[1], vs[0], vs[1]));
      }
      for (PetscInt xi = xs[0] - 1; xi < xs[0] + 3; ++xi) {
        // Treat xi as periodic
        const PetscInt xip = xi < 0 ? xi + xend[0] : (xi >= xend[0] ? xi - xend[0] : xi);
        PetscCall(W_3_Interpolation_Private((xmin[0] + (xi + 0.5) * hx[0] - x[p * dim + 0]) / hx[0], &Wx[0]));
        for (PetscInt xj = PetscMax(xs[1] - 1, 0); xj < PetscMin(xs[1] + 3, xend[1]); ++xj) {
          if (xcdim > 1) PetscCall(W_3_Interpolation_Private((xmin[1] + (xj + 0.5) * hx[1] - x[p * dim + 1]) / hx[1], &Wx[1]));
          else Wx[1] = 1.;
          for (PetscInt vi = PetscMax(vs[0] - 1, 0); vi < PetscMin(vs[0] + 3, vend[0]); ++vi) {
            PetscCall(W_3_Interpolation_Private((vmin[0] + (vi + 0.5) * hv[0] - v[p * dim + 0]) / hv[0], &Wv[0]));
            for (PetscInt vj = PetscMax(vs[1] - 1, 0); vj < PetscMin(vs[1] + 3, vend[1]); ++vj) {
              const PetscInt rc = xip * xend[1] + xj;
              const PetscInt rv = vi * vend[1] + vj;

              PetscCall(DMSwarmSortGetPointsPerCell(*rsw, rc, &rNpc, &rpidx));
              if (vcdim > 1) PetscCall(W_3_Interpolation_Private((vmin[1] + (vj + 0.5) * hv[1] - v[p * dim + 1]) / hv[1], &Wv[1]));
              else Wv[1] = 1.;
              if (debug > 2)
                PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Depositing on particle (%" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ", %" PetscInt_FMT ") w = %g (%g, %g, %g, %g)\n", xi, xj, vi, vj, (double)(wp * Wx[0] * Wx[1] * Wv[0] * Wv[1]), (double)Wx[0], (double)Wx[1], (double)Wv[0], (double)Wv[1]));
              // Add weight to new particles from original particle using interpolation function
              PetscCheck(rNpc == vend[0] * vend[1], PETSC_COMM_SELF, PETSC_ERR_PLIB, "Invalid particle velocity binning");
              const PetscInt rp = rpidx[rv];
              PetscCheck(rp >= 0 && rp < Np, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Particle index %" PetscInt_FMT " not in [0, %" PetscInt_FMT ")", rp, Np);
              rw[rp] += wp * Wx[0] * Wx[1] * Wv[0] * Wv[1];
              if (debug > 2) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Adding weight %g (%g) to particle %" PetscInt_FMT "\n", (double)(wp * Wx[0] * Wx[1] * Wv[0] * Wv[1]), (double)PetscRealPart(rw[rp]), rp));
              PetscCall(DMSwarmSortRestorePointsPerCell(*rsw, rc, &rNpc, &rpidx));
            }
          }
        }
      }
    }
    PetscCall(DMSwarmSortRestorePointsPerCell(sw, c, &Npc, &pidx));
  }
  PetscCall(DMSwarmSortRestoreAccess(sw));
  PetscCall(DMSwarmSortRestoreAccess(*rsw));
  PetscCall(DMSwarmRestoreField(sw, coordFields[0], NULL, NULL, (void **)&x));
  PetscCall(DMSwarmRestoreField(sw, "velocity", NULL, NULL, (void **)&v));
  PetscCall(DMSwarmRestoreField(sw, "w_q", NULL, NULL, (void **)&w));
  PetscCall(DMSwarmRestoreField(*rsw, "w_q", NULL, NULL, (void **)&rw));

  if (debug) {
    Vec w;

    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, coordFields[0], &w));
    PetscCall(VecViewFromOptions(w, NULL, "-remap_view"));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, coordFields[0], &w));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "velocity", &w));
    PetscCall(VecViewFromOptions(w, NULL, "-remap_view"));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "velocity", &w));
    PetscCall(DMSwarmCreateGlobalVectorFromField(sw, "w_q", &w));
    PetscCall(VecViewFromOptions(w, NULL, "-remap_view"));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, "w_q", &w));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*rsw, coordFields[0], &w));
    PetscCall(VecViewFromOptions(w, NULL, "-remap_view"));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*rsw, coordFields[0], &w));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*rsw, "velocity", &w));
    PetscCall(VecViewFromOptions(w, NULL, "-remap_view"));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*rsw, "velocity", &w));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*rsw, "w_q", &w));
    PetscCall(VecViewFromOptions(w, NULL, "-remap_view"));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*rsw, "w_q", &w));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static void f0_v2(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f0[])
{
  PetscInt d;

  f0[0] = 0.0;
  for (d = dim / 2; d < dim; ++d) f0[0] += PetscSqr(x[d]) * u[0];
}

static PetscErrorCode DMSwarmRemap_PFAK_Internal(DM sw, DM *rsw)
{
  DM            xdm, vdm, rdm;
  DMSwarmCellDM rcelldm;
  Mat           M_p, rM_p, rPM_p;
  Vec           w, rw, rhs;
  PetscInt      Nf;
  const char  **fields;

  PetscFunctionBegin;
  // Create a new centroid swarm without weights
  PetscCall(DMSwarmGetCellDM(sw, &xdm));
  PetscCall(DMSwarmSetCellDMActive(sw, "velocity"));
  PetscCall(DMSwarmGetCellDMActive(sw, &rcelldm));
  PetscCall(DMSwarmCellDMGetDM(rcelldm, &vdm));
  PetscCall(DMSwarmDuplicate(sw, rsw));
  // Set remap cell DM
  PetscCall(DMSwarmSetCellDMActive(sw, "remap"));
  PetscCall(DMSwarmGetCellDMActive(sw, &rcelldm));
  PetscCall(DMSwarmCellDMGetFields(rcelldm, &Nf, &fields));
  PetscCheck(Nf == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_ARG_WRONG, "We only allow a single weight field, not %" PetscInt_FMT, Nf);
  PetscCall(DMSwarmGetCellDM(sw, &rdm));
  PetscCall(DMGetGlobalVector(rdm, &rhs));
  PetscCall(DMSwarmMigrate(sw, PETSC_FALSE)); // Bin particles in remap mesh
  // Compute rhs = M_p w_p
  PetscCall(DMCreateMassMatrix(sw, rdm, &M_p));
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, fields[0], &w));
  PetscCall(VecViewFromOptions(w, NULL, "-remap_w_view"));
  PetscCall(MatMultTranspose(M_p, w, rhs));
  PetscCall(VecViewFromOptions(rhs, NULL, "-remap_rhs_view"));
  PetscCall(DMSwarmDestroyGlobalVectorFromField(sw, fields[0], &w));
  PetscCall(MatDestroy(&M_p));
  {
    KSP         ksp;
    Mat         M_f;
    Vec         u_f;
    PetscReal   mom[4];
    PetscInt    cdim;
    const char *prefix;

    PetscCall(DMGetCoordinateDim(rdm, &cdim));
    PetscCall(DMCreateMassMatrix(rdm, rdm, &M_f));
    PetscCall(DMGetGlobalVector(rdm, &u_f));

    PetscCall(KSPCreate(PetscObjectComm((PetscObject)sw), &ksp));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)sw, &prefix));
    PetscCall(KSPSetOptionsPrefix(ksp, prefix));
    PetscCall(KSPAppendOptionsPrefix(ksp, "ptof_"));
    PetscCall(KSPSetFromOptions(ksp));

    PetscCall(KSPSetOperators(ksp, M_f, M_f));
    PetscCall(KSPSolve(ksp, rhs, u_f));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(VecViewFromOptions(u_f, NULL, "-remap_uf_view"));

    PetscCall(DMPlexComputeMoments(rdm, u_f, mom));
    // Energy is not correct since it uses (x^2 + v^2)
    PetscDS     rds;
    PetscScalar rmom;
    void       *ctx;

    PetscCall(DMGetDS(rdm, &rds));
    PetscCall(DMGetApplicationContext(rdm, &ctx));
    PetscCall(PetscDSSetObjective(rds, 0, &f0_v2));
    PetscCall(DMPlexComputeIntegralFEM(rdm, u_f, &rmom, ctx));
    mom[1 + cdim] = PetscRealPart(rmom);

    PetscCall(DMRestoreGlobalVector(rdm, &u_f));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "========== PFAK u_f ==========\n"));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Mom 0: %g\n", (double)mom[0]));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Mom x: %g\n", (double)mom[1 + 0]));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Mom v: %g\n", (double)mom[1 + 1]));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "Mom 2: %g\n", (double)mom[1 + cdim]));
    PetscCall(MatDestroy(&M_f));
  }
  // Create Remap particle mass matrix M_p
  PetscInt xcStart, xcEnd, vcStart, vcEnd, cStart, cEnd, r;

  PetscCall(DMSwarmSetCellDMActive(*rsw, "remap"));
  PetscCall(DMPlexGetHeightStratum(xdm, 0, &xcStart, &xcEnd));
  PetscCall(DMPlexGetHeightStratum(vdm, 0, &vcStart, &vcEnd));
  PetscCall(DMPlexGetHeightStratum(rdm, 0, &cStart, &cEnd));
  r = (PetscInt)PetscSqrtReal(((xcEnd - xcStart) * (vcEnd - vcStart)) / (cEnd - cStart));
  PetscCall(InitializeParticles_Regular(*rsw, r));
  PetscCall(DMSwarmMigrate(*rsw, PETSC_FALSE)); // Bin particles in remap mesh
  PetscCall(DMCreateMassMatrix(*rsw, rdm, &rM_p));
  PetscCall(MatViewFromOptions(rM_p, NULL, "-rM_p_view"));
  // Solve M_p
  {
    KSP         ksp;
    PC          pc;
    const char *prefix;
    PetscBool   isBjacobi;

    PetscCall(KSPCreate(PetscObjectComm((PetscObject)sw), &ksp));
    PetscCall(PetscObjectGetOptionsPrefix((PetscObject)sw, &prefix));
    PetscCall(KSPSetOptionsPrefix(ksp, prefix));
    PetscCall(KSPAppendOptionsPrefix(ksp, "ftop_"));
    PetscCall(KSPSetFromOptions(ksp));

    PetscCall(KSPGetPC(ksp, &pc));
    PetscCall(PetscObjectTypeCompare((PetscObject)pc, PCBJACOBI, &isBjacobi));
    if (isBjacobi) {
      PetscCall(DMSwarmCreateMassMatrixSquare(sw, rdm, &rPM_p));
    } else {
      rPM_p = rM_p;
      PetscCall(PetscObjectReference((PetscObject)rPM_p));
    }
    PetscCall(KSPSetOperators(ksp, rM_p, rPM_p));
    PetscCall(DMSwarmCreateGlobalVectorFromField(*rsw, fields[0], &rw));
    PetscCall(KSPSolveTranspose(ksp, rhs, rw));
    PetscCall(VecViewFromOptions(rw, NULL, "-remap_rw_view"));
    PetscCall(DMSwarmDestroyGlobalVectorFromField(*rsw, fields[0], &rw));
    PetscCall(KSPDestroy(&ksp));
    PetscCall(MatDestroy(&rPM_p));
    PetscCall(MatDestroy(&rM_p));
  }
  PetscCall(DMRestoreGlobalVector(rdm, &rhs));

  // Restore original cell DM
  PetscCall(DMSwarmSetCellDMActive(sw, "space"));
  PetscCall(DMSwarmSetCellDMActive(*rsw, "space"));
  PetscCall(DMSwarmMigrate(*rsw, PETSC_FALSE)); // Bin particles in spatial mesh
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMSwarmRemapMonitor_Internal(DM sw, DM rsw)
{
  PetscReal mom[4], rmom[4];
  PetscInt  cdim;

  PetscFunctionBegin;
  PetscCall(DMGetCoordinateDim(sw, &cdim));
  PetscCall(DMSwarmComputeMoments(sw, "velocity", "w_q", mom));
  PetscCall(DMSwarmComputeMoments(rsw, "velocity", "w_q", rmom));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "========== Remapped ==========\n"));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Mom 0: %g --> %g\n", (double)mom[0], (double)rmom[0]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Mom 1: %g --> %g\n", (double)mom[1], (double)rmom[1]));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Mom 2: %g --> %g\n", (double)mom[1 + cdim], (double)rmom[1 + cdim]));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMSwarmRemap - Project the swarm fields onto a new set of particles

  Collective

  Input Parameter:
. sw - The `DMSWARM` object

  Level: beginner

.seealso: [](ch_dmbase), `DMSWARM`, `DMSwarmMigrate()`, `DMSwarmCrate()`
@*/
PetscErrorCode DMSwarmRemap(DM sw)
{
  DM_Swarm *swarm = (DM_Swarm *)sw->data;
  DM        rsw;

  PetscFunctionBegin;
  switch (swarm->remap_type) {
  case DMSWARM_REMAP_NONE:
    PetscFunctionReturn(PETSC_SUCCESS);
  case DMSWARM_REMAP_COLELLA:
    PetscCall(DMSwarmRemap_Colella_Internal(sw, &rsw));
    break;
  case DMSWARM_REMAP_PFAK:
    PetscCall(DMSwarmRemap_PFAK_Internal(sw, &rsw));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "No remap algorithm %s", DMSwarmRemapTypeNames[swarm->remap_type]);
  }
  PetscCall(DMSwarmRemapMonitor_Internal(sw, rsw));
  PetscCall(DMSwarmReplace(sw, &rsw));
  PetscFunctionReturn(PETSC_SUCCESS);
}
