#include <petsc/private/dmimpl.h>
#include <petscdm.h>      /*I "petscdm.h" I*/
#include <petscdmda.h>    /*I "petscdmda.h" I*/
#include <petscdmplex.h>  /*I "petscdmplex.h" I*/
#include <petscdmswarm.h> /*I "petscdmswarm.h" I*/
#include <petscksp.h>     /*I "petscksp.h" I*/
#include <petscblaslapack.h>

#include <petsc/private/dmswarmimpl.h>         // For the citation and check
#include "../src/dm/impls/swarm/data_bucket.h" // For DataBucket internals

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
- y  - The global vector: the input value of globalVec is used as an initial guess

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
  PetscCall(MatShellSetOperation(CtC, MATOP_MULT, (void (*)(void))MatMult_GlobalToLocalNormal));
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
. funcs - The functions to evaluate, one per field
- mode  - The insertion mode for values

  Output Parameter:
. X - The output vector

  Calling sequence of `funcs`:
+ dim          - The spatial dimension
. Nf           - The number of input fields
. NfAux        - The number of input auxiliary fields
. uOff         - The offset of each field in `u`
. uOff_x       - The offset of each field in `u_x`
. u            - The field values at this point in space
. u_t          - The field time derivative at this point in space (or `NULL`)
. u_x          - The field derivatives at this point in space
. aOff         - The offset of each auxiliary field in `u`
. aOff_x       - The offset of each auxiliary field in `u_x`
. a            - The auxiliary field values at this point in space
. a_t          - The auxiliary field time derivative at this point in space (or `NULL`)
. a_x          - The auxiliary field derivatives at this point in space
. t            - The current time
. x            - The coordinates of this point
. numConstants - The number of constants
. constants    - The value of each constant
- f            - The value of the function at this point in space

  Level: advanced

  Note:
  There are three different `DM`s that potentially interact in this function. The output `dm`, specifies the layout of the values calculates by the function.
  The input `DM`, attached to `U`, may be different. For example, you can input the solution over the full domain, but output over a piece of the boundary, or
  a subdomain. You can also output a different number of fields than the input, with different discretizations. Last the auxiliary `DM`, attached to the
  auxiliary field vector, which is attached to `dm`, can also be different. It can have a different topology, number of fields, and discretizations.

.seealso: [](ch_dmbase), `DM`, `DMProjectFieldLocal()`, `DMProjectFieldLabelLocal()`, `DMProjectFunction()`, `DMComputeL2Diff()`
@*/
PetscErrorCode DMProjectField(DM dm, PetscReal time, Vec U, void (**funcs)(PetscInt dim, PetscInt Nf, PetscInt NfAux, const PetscInt uOff[], const PetscInt uOff_x[], const PetscScalar u[], const PetscScalar u_t[], const PetscScalar u_x[], const PetscInt aOff[], const PetscInt aOff_x[], const PetscScalar a[], const PetscScalar a_t[], const PetscScalar a_x[], PetscReal t, const PetscReal x[], PetscInt numConstants, const PetscScalar constants[], PetscScalar f[]), InsertMode mode, Vec X)
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

  PetscCall(MatMatMult(globalA, MF, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &AF));
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
    PetscCall(MatTransposeMatMult(In, MF, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &MC));
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
  PetscCall(MatMultTranspose(M_f, u_f, rhs));

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
  Vec u;

  PetscFunctionBegin;
  PetscCall(PetscCitationsRegister(SwarmProjCitation, &SwarmProjcite));
  PetscCheck(Nf == 1, PetscObjectComm((PetscObject)sw), PETSC_ERR_SUP, "Currently supported only for a single field");
  PetscCall(DMSwarmCreateGlobalVectorFromField(sw, fieldnames[0], &u));
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
  Vec                v_field_l, denom_l, coor_l, denom;
  PetscScalar       *_field_l, *_denom_l;
  PetscInt           k, p, e, npoints, nel, npe;
  PetscInt          *mpfield_cell;
  PetscReal         *mpfield_coor;
  const PetscInt    *element_list;
  const PetscInt    *element;
  PetscScalar        xi_p[2], Ni[4];
  const PetscScalar *_coor;

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

  PetscCall(DMDAGetElements(dm, &nel, &npe, &element_list));
  PetscCall(DMSwarmGetLocalSize(swarm, &npoints));
  PetscCall(DMSwarmGetField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&mpfield_coor));
  PetscCall(DMSwarmGetField(swarm, DMSwarmPICField_cellid, NULL, NULL, (void **)&mpfield_cell));

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

  PetscCall(DMSwarmRestoreField(swarm, DMSwarmPICField_cellid, NULL, NULL, (void **)&mpfield_cell));
  PetscCall(DMSwarmRestoreField(swarm, DMSwarmPICField_coor, NULL, NULL, (void **)&mpfield_coor));
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
  DMSwarmProjectFields - Project a set of swarm fields onto the cell `DM`

  Collective

  Input Parameters:
+ dm         - the `DMSWARM`
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
PetscErrorCode DMSwarmProjectFields(DM dm, PetscInt nfields, const char *fieldnames[], Vec fields[], ScatterMode mode)
{
  DM_Swarm         *swarm = (DM_Swarm *)dm->data;
  DMSwarmDataField *gfield;
  DM                celldm;
  PetscBool         isDA, isPlex;
  MPI_Comm          comm;

  PetscFunctionBegin;
  DMSWARMPICVALID(dm);
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMSwarmGetCellDM(dm, &celldm));
  PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMDA, &isDA));
  PetscCall(PetscObjectTypeCompare((PetscObject)celldm, DMPLEX, &isPlex));
  PetscCall(PetscMalloc1(nfields, &gfield));
  for (PetscInt f = 0; f < nfields; ++f) PetscCall(DMSwarmDataBucketGetDMSwarmDataFieldByName(swarm->db, fieldnames[f], &gfield[f]));

  if (isDA) {
    for (PetscInt f = 0; f < nfields; f++) {
      PetscCheck(gfield[f]->petsc_type == PETSC_REAL, comm, PETSC_ERR_SUP, "Projection only valid for fields using a data type = PETSC_REAL");
      PetscCheck(gfield[f]->bs == 1, comm, PETSC_ERR_SUP, "Projection only valid for fields with block size = 1");
    }
    PetscCall(DMSwarmProjectFields_DA_Internal(dm, celldm, nfields, gfield, fields, mode));
  } else if (isPlex) {
    PetscInt Nf;

    PetscCall(DMGetNumFields(celldm, &Nf));
    PetscCheck(Nf == nfields, comm, PETSC_ERR_ARG_WRONG, "Number of DM fields %" PetscInt_FMT " != %" PetscInt_FMT " number of requested Swarm fields", Nf, nfields);
    PetscCall(DMSwarmProjectFields_Plex_Internal(dm, celldm, nfields, fieldnames, fields[0], mode));
  } else SETERRQ(PetscObjectComm((PetscObject)dm), PETSC_ERR_SUP, "Only supported for cell DMs of type DMDA and DMPLEX");

  PetscCall(PetscFree(gfield));
  PetscFunctionReturn(PETSC_SUCCESS);
}
