/*
            This implements Richardson Iteration.
*/
#include <../src/ksp/ksp/impls/rich/richardsonimpl.h> /*I "petscksp.h" I*/

static PetscErrorCode KSPSetUp_Richardson(KSP ksp)
{
  KSP_Richardson *richardsonP = (KSP_Richardson *)ksp->data;

  PetscFunctionBegin;
  if (richardsonP->selfscale) {
    PetscCall(KSPSetWorkVecs(ksp, 4));
  } else {
    PetscCall(KSPSetWorkVecs(ksp, 2));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_Richardson(KSP ksp)
{
  PetscReal       rnorm = 0.0, abr;
  PetscScalar     scale, rdot;
  Vec             x, b, r, z, w = NULL, y = NULL;
  PetscInt        i, maxit, xs, ws;
  Mat             Amat, Pmat;
  KSP_Richardson *richardsonP = (KSP_Richardson *)ksp->data;
  PetscBool       exists, diagonalscale;
  MatNullSpace    nullsp;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

  ksp->its = 0;

  PetscCall(PCGetOperators(ksp->pc, &Amat, &Pmat));
  x = ksp->vec_sol;
  b = ksp->vec_rhs;
  PetscCall(VecGetSize(x, &xs));
  PetscCall(VecGetSize(ksp->work[0], &ws));
  if (xs != ws) {
    if (richardsonP->selfscale) {
      PetscCall(KSPSetWorkVecs(ksp, 4));
    } else {
      PetscCall(KSPSetWorkVecs(ksp, 2));
    }
  }
  r = ksp->work[0];
  z = ksp->work[1];
  if (richardsonP->selfscale) {
    w = ksp->work[2];
    y = ksp->work[3];
  }
  maxit = ksp->max_it;

  /* if user has provided fast Richardson code use that */
  PetscCall(PCApplyRichardsonExists(ksp->pc, &exists));
  PetscCall(MatGetNullSpace(Pmat, &nullsp));
  if (exists && maxit > 0 && richardsonP->scale == 1.0 && (ksp->converged == KSPConvergedDefault || ksp->converged == KSPConvergedSkip) && !ksp->numbermonitors && !ksp->transpose_solve && !nullsp) {
    PCRichardsonConvergedReason reason;
    PetscCall(PCApplyRichardson(ksp->pc, b, x, r, ksp->rtol, ksp->abstol, ksp->divtol, maxit, ksp->guess_zero, &ksp->its, &reason));
    ksp->reason = (KSPConvergedReason)reason;
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  if (!ksp->guess_zero) { /*   r <- b - A x     */
    PetscCall(KSP_MatMult(ksp, Amat, x, r));
    PetscCall(VecAYPX(r, -1.0, b));
  } else {
    PetscCall(VecCopy(b, r));
  }

  ksp->its = 0;
  if (richardsonP->selfscale) {
    PetscCall(KSP_PCApply(ksp, r, z)); /*   z <- B r          */
    for (i = 0; i < maxit; i++) {
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
        PetscCall(VecNorm(r, NORM_2, &rnorm)); /*   rnorm <- r'*r     */
      } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
        PetscCall(VecNorm(z, NORM_2, &rnorm)); /*   rnorm <- z'*z     */
      } else rnorm = 0.0;

      KSPCheckNorm(ksp, rnorm);
      ksp->rnorm = rnorm;
      PetscCall(KSPMonitor(ksp, i, rnorm));
      PetscCall(KSPLogResidualHistory(ksp, rnorm));
      PetscCall((*ksp->converged)(ksp, i, rnorm, &ksp->reason, ksp->cnvP));
      if (ksp->reason) break;
      PetscCall(KSP_PCApplyBAorAB(ksp, z, y, w)); /* y = BAz = BABr */
      PetscCall(VecDotNorm2(z, y, &rdot, &abr));  /*   rdot = (Br)^T(BABR); abr = (BABr)^T (BABr) */
      scale = rdot / abr;
      PetscCall(PetscInfo(ksp, "Self-scale factor %g\n", (double)PetscRealPart(scale)));
      PetscCall(VecAXPY(x, scale, z));  /*   x  <- x + scale z */
      PetscCall(VecAXPY(r, -scale, w)); /*  r <- r - scale*Az */
      PetscCall(VecAXPY(z, -scale, y)); /*  z <- z - scale*y */
      ksp->its++;
    }
  } else {
    for (i = 0; i < maxit; i++) {
      if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
        PetscCall(VecNorm(r, NORM_2, &rnorm)); /*   rnorm <- r'*r     */
      } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
        PetscCall(KSP_PCApply(ksp, r, z));     /*   z <- B r          */
        PetscCall(VecNorm(z, NORM_2, &rnorm)); /*   rnorm <- z'*z     */
      } else rnorm = 0.0;
      ksp->rnorm = rnorm;
      PetscCall(KSPMonitor(ksp, i, rnorm));
      PetscCall(KSPLogResidualHistory(ksp, rnorm));
      PetscCall((*ksp->converged)(ksp, i, rnorm, &ksp->reason, ksp->cnvP));
      if (ksp->reason) break;
      if (ksp->normtype != KSP_NORM_PRECONDITIONED) PetscCall(KSP_PCApply(ksp, r, z)); /*   z <- B r          */

      PetscCall(VecAXPY(x, richardsonP->scale, z)); /*   x  <- x + scale z */
      ksp->its++;

      if (i + 1 < maxit || ksp->normtype != KSP_NORM_NONE) {
        PetscCall(KSP_MatMult(ksp, Amat, x, r)); /*   r  <- b - Ax      */
        PetscCall(VecAYPX(r, -1.0, b));
      }
    }
  }
  if (!ksp->reason) {
    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      PetscCall(VecNorm(r, NORM_2, &rnorm)); /*   rnorm <- r'*r     */
    } else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
      PetscCall(KSP_PCApply(ksp, r, z));     /*   z <- B r          */
      PetscCall(VecNorm(z, NORM_2, &rnorm)); /*   rnorm <- z'*z     */
    } else rnorm = 0.0;

    KSPCheckNorm(ksp, rnorm);
    ksp->rnorm = rnorm;
    PetscCall(KSPLogResidualHistory(ksp, rnorm));
    PetscCall(KSPMonitor(ksp, i, rnorm));
    if (ksp->its >= ksp->max_it) {
      if (ksp->normtype != KSP_NORM_NONE) {
        PetscCall((*ksp->converged)(ksp, i, rnorm, &ksp->reason, ksp->cnvP));
        if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
      } else {
        ksp->reason = KSP_CONVERGED_ITS;
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Replacement for KSPCheckNorm() in KSPMatSolve_Richardson(), where ksp->vec_sol is not available since the solution is a block of vectors
*/
static PetscErrorCode KSPMatSolveCheckNorm_Richardson(KSP ksp, PetscReal rnorm, Mat X)
{
  PCFailedReason pcreason;

  PetscFunctionBegin;
  if (!PetscIsInfOrNanReal(rnorm)) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCheck(!ksp->errorifnotconverged, PetscObjectComm((PetscObject)ksp), PETSC_ERR_NOT_CONVERGED, "KSPMatSolve%s() has not converged due to infinity or NaN norm", ksp->transpose.solve_requested ? "Transpose" : "");
  PetscCall(PCReduceFailedReason(ksp->pc));
  PetscCall(PCGetFailedReason(ksp->pc, &pcreason));
  /* as with VecFlag() in KSPCheckNorm(), the state of the block of solutions increases exactly once whether or not it is flagged, so that an outer solver detects the failure, MatSetInf() does the increase itself and PCReduceFailedReason() above makes pcreason the same on all processes */
  if (pcreason) PetscCall(MatSetInf(X));
  else PetscCall(PetscObjectStateIncrease((PetscObject)X));
  ksp->reason = pcreason ? KSP_DIVERGED_PC_FAILED : KSP_DIVERGED_NANORINF;
  ksp->rnorm  = rnorm;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
   Block analog of KSP_PCApply(), the null space of Amat is removed from each column of the block of preconditioned vectors as in KSP_RemoveNullSpaceMat()
*/
static PetscErrorCode KSPMatSolvePCMatApply_Richardson(KSP ksp, Mat R, Mat Z)
{
  PetscFunctionBegin;
  PetscCall(KSP_PCMatApply(ksp, R, Z));
  PetscCall(KSP_RemoveNullSpaceMat(ksp, Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPMatSolve_Richardson(KSP ksp, Mat B, Mat X)
{
  PetscReal       rnorm = 0.0;
  Mat             Amat, Pmat, R, Z;
  Vec             cb, cx, w;
  PetscInt        i, maxit, N;
  KSP_Richardson *richardsonP = (KSP_Richardson *)ksp->data;
  PetscBool       diagonalscale, exists, matexists;
  MatNullSpace    nullsp;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

  ksp->its    = 0;
  ksp->reason = KSP_CONVERGED_ITERATING;
  maxit       = ksp->max_it;
  PetscCall(PCGetOperators(ksp->pc, &Amat, &Pmat));

  /* if user has provided fast Richardson code use that, with the same conditions as in KSPSolve_Richardson() except for the monitors, which are not called during KSPMatSolve() */
  PetscCall(PCApplyRichardsonExists(ksp->pc, &exists));
  PetscCall(PCMatApplyRichardsonExists(ksp->pc, &matexists));
  PetscCall(MatGetNullSpace(Pmat, &nullsp));
  if ((exists || matexists) && maxit > 0 && richardsonP->scale == 1.0 && (ksp->converged == KSPConvergedDefault || ksp->converged == KSPConvergedSkip) && !ksp->transpose_solve && !nullsp) {
    PCRichardsonConvergedReason reason;

    if (matexists) {
      PetscCall(PetscInfo(ksp, "Using PCMatApplyRichardson() on each batch of right-hand sides, by default the whole block\n"));
      PetscCall(PCMatApplyRichardson(ksp->pc, B, X, NULL, ksp->rtol, ksp->abstol, ksp->divtol, maxit, ksp->guess_zero, &ksp->its, &reason));
      ksp->reason = (KSPConvergedReason)reason;
    } else {
      /* the block iteration below does not compute the same thing as PCApplyRichardson(), so the right-hand sides are solved one at a time to match KSPSolve() */
      PetscCall(PetscInfo(ksp, "Using PCApplyRichardson() on one right-hand side at a time\n"));
      PetscCall(MatGetSize(B, NULL, &N));
      /* ksp->work may have the size of an earlier operator since setup is not run again, so a fresh work vector is created */
      PetscCall(MatCreateVecs(B, NULL, &w));
      /* as with the column-by-column fallback of KSPMatSolve_Private(), ksp->reason and ksp->its reflect the last right-hand side */
      for (i = 0; i < N; i++) {
        PetscCall(MatDenseGetColumnVecRead(B, i, &cb));
        if (ksp->guess_zero) PetscCall(MatDenseGetColumnVecWrite(X, i, &cx));
        else PetscCall(MatDenseGetColumnVec(X, i, &cx));
        PetscCall(PCApplyRichardson(ksp->pc, cb, cx, w, ksp->rtol, ksp->abstol, ksp->divtol, maxit, ksp->guess_zero, &ksp->its, &reason));
        if (ksp->guess_zero) PetscCall(MatDenseRestoreColumnVecWrite(X, i, &cx));
        else PetscCall(MatDenseRestoreColumnVec(X, i, &cx));
        PetscCall(MatDenseRestoreColumnVecRead(B, i, &cb));
        ksp->reason = (KSPConvergedReason)reason;
      }
      PetscCall(VecDestroy(&w));
    }
    PetscFunctionReturn(PETSC_SUCCESS);
  }

  PetscCall(PetscInfo(ksp, "Iterating on each batch of right-hand sides, by default the whole block\n"));
  /* R and Z are rebuilt on each call since the number of right-hand sides may change between calls */
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &R));
  PetscCall(MatDuplicate(B, MAT_DO_NOT_COPY_VALUES, &Z));
  /* set the product A X (or A^T X) up once so that only its numeric phase is run in the loop below */
  PetscCall(MatProductCreateWithMat(Amat, X, NULL, R));
  PetscCall(MatProductSetType(R, ksp->transpose_solve ? MATPRODUCT_AtB : MATPRODUCT_AB));
  PetscCall(MatProductSetFromOptions(R));
  PetscCall(MatProductSymbolic(R));

  if (!ksp->guess_zero) { /*   R <- B - A X      */
    PetscCall(MatProductNumeric(R));
    PetscCall(MatAYPX(R, -1.0, B, SAME_NONZERO_PATTERN));
  } else PetscCall(MatCopy(B, R, SAME_NONZERO_PATTERN));

  for (i = 0; i < maxit; i++) {
    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) PetscCall(MatNorm(R, NORM_FROBENIUS, &rnorm)); /*   rnorm <- ||R||_F  */
    else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
      PetscCall(KSPMatSolvePCMatApply_Richardson(ksp, R, Z)); /*   Z <- B R          */
      PetscCall(MatNorm(Z, NORM_FROBENIUS, &rnorm));          /*   rnorm <- ||Z||_F  */
    } else rnorm = 0.0;

    PetscCall(KSPMatSolveCheckNorm_Richardson(ksp, rnorm, X));
    if (ksp->reason) break;
    ksp->rnorm = rnorm;
    PetscCall(KSPLogResidualHistory(ksp, rnorm));
    PetscCall((*ksp->converged)(ksp, i, rnorm, &ksp->reason, ksp->cnvP));
    if (ksp->reason) break;
    if (ksp->normtype != KSP_NORM_PRECONDITIONED) PetscCall(KSPMatSolvePCMatApply_Richardson(ksp, R, Z)); /*   Z <- B R          */

    PetscCall(MatAXPY(X, richardsonP->scale, Z, SAME_NONZERO_PATTERN)); /*   X  <- X + scale Z */
    ksp->its++;

    if (i + 1 < maxit || ksp->normtype != KSP_NORM_NONE) {
      PetscCall(MatProductNumeric(R)); /*   R  <- B - A X     */
      PetscCall(MatAYPX(R, -1.0, B, SAME_NONZERO_PATTERN));
    }
  }
  if (!ksp->reason) {
    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) PetscCall(MatNorm(R, NORM_FROBENIUS, &rnorm)); /*   rnorm <- ||R||_F  */
    else if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
      PetscCall(KSPMatSolvePCMatApply_Richardson(ksp, R, Z)); /*   Z <- B R          */
      PetscCall(MatNorm(Z, NORM_FROBENIUS, &rnorm));          /*   rnorm <- ||Z||_F  */
    } else rnorm = 0.0;

    PetscCall(KSPMatSolveCheckNorm_Richardson(ksp, rnorm, X));
    if (!ksp->reason) {
      ksp->rnorm = rnorm;
      PetscCall(KSPLogResidualHistory(ksp, rnorm));
      if (ksp->its >= ksp->max_it) {
        if (ksp->normtype != KSP_NORM_NONE) {
          PetscCall((*ksp->converged)(ksp, i, rnorm, &ksp->reason, ksp->cnvP));
          if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
        } else ksp->reason = KSP_CONVERGED_ITS;
      }
    }
  }
  PetscCall(MatProductClear(R));
  PetscCall(MatDestroy(&R));
  PetscCall(MatDestroy(&Z));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPView_Richardson(KSP ksp, PetscViewer viewer)
{
  KSP_Richardson *richardsonP = (KSP_Richardson *)ksp->data;
  PetscBool       isascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    if (richardsonP->selfscale) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  using self-scale best computed damping factor\n"));
    } else {
      PetscCall(PetscViewerASCIIPrintf(viewer, "  damping factor=%g\n", (double)richardsonP->scale));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetFromOptions_Richardson(KSP ksp, PetscOptionItems PetscOptionsObject)
{
  KSP_Richardson *rich = (KSP_Richardson *)ksp->data;
  PetscReal       tmp;
  PetscBool       flg, flg2;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP Richardson Options");
  PetscCall(PetscOptionsReal("-ksp_richardson_scale", "damping factor", "KSPRichardsonSetScale", rich->scale, &tmp, &flg));
  if (flg) PetscCall(KSPRichardsonSetScale(ksp, tmp));
  PetscCall(PetscOptionsBool("-ksp_richardson_self_scale", "dynamically determine optimal damping factor", "KSPRichardsonSetSelfScale", rich->selfscale, &flg2, &flg));
  if (flg) PetscCall(KSPRichardsonSetSelfScale(ksp, flg2));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPDestroy_Richardson(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPRichardsonSetScale_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPRichardsonSetSelfScale_C", NULL));
  PetscCall(KSPDestroyDefault(ksp));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPRichardsonSetScale_Richardson(KSP ksp, PetscReal scale)
{
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  richardsonP        = (KSP_Richardson *)ksp->data;
  richardsonP->scale = scale;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPRichardsonSetSelfScale_Richardson(KSP ksp, PetscBool selfscale)
{
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  richardsonP = (KSP_Richardson *)ksp->data;
  /* KSPSetUp_Richardson() picks the number of work vectors from this flag, so setup must be run again */
  if (richardsonP->selfscale != selfscale) ksp->setupstage = KSP_SETUP_NEW;
  richardsonP->selfscale = selfscale;
  /* the self-scaled variant has no block analog, so KSPMatSolve() falls back to solving the right-hand sides one at a time */
  ksp->ops->matsolve = selfscale ? NULL : KSPMatSolve_Richardson;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPBuildResidual_Richardson(KSP ksp, Vec t, Vec v, Vec *V)
{
  PetscFunctionBegin;
  if (ksp->normtype == KSP_NORM_NONE) {
    PetscCall(KSPBuildResidualDefault(ksp, t, v, V));
  } else {
    PetscCall(VecCopy(ksp->work[0], v));
    *V = v;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
    KSPRICHARDSON - The preconditioned Richardson iterative method {cite}`richarson1911`

   Options Database Key:
.   -ksp_richardson_scale - damping factor on the correction (defaults to 1.0)

   Level: beginner

   Notes:
   $ x^{n+1} = x^{n} + scale*B(b - A x^{n})$

   Here B is the application of the preconditioner

   This method often (usually) will not converge unless scale is very small.

   For some preconditioners, currently `PCSOR`, the convergence test is skipped to improve speed,
   thus it always iterates the maximum number of iterations you've selected. When -ksp_monitor
   (or any other monitor) is turned on, the norm is computed at each iteration and so the convergence test is run unless
   you specifically call `KSPSetNormType`(ksp,`KSP_NORM_NONE`);

   For some preconditioners, currently `PCMG` and `PCHYPRE` with BoomerAMG if -ksp_monitor (and also
   any other monitor) is not turned on then the convergence test is done by the preconditioner itself and
   so the solver may run more or fewer iterations then if -ksp_monitor is selected.

   Supports only left preconditioning

   If using direct solvers such as `PCLU` and `PCCHOLESKY` one generally uses `KSPPREONLY` instead of this which uses exactly one iteration

   `-ksp_type richardson -pc_type jacobi` gives one classical Jacobi preconditioning

   `KSPMatSolve()` and `KSPMatSolveTranspose()` are supported natively, that is, the iteration is performed on a whole batch of right-hand sides at once, a batch being
   the whole block unless `KSPSetMatSolveBatchSize()` is used, except with `KSPRichardsonSetSelfScale()`, for which the right-hand sides are solved one at a time.
   The convergence of each batch is tested on the Frobenius norm of its block of (preconditioned) residuals and, with a nonzero initial guess, the relative tolerance
   is by default based on the Frobenius norm of its block of (preconditioned) right-hand sides, as in `KSPSolve()`. `KSPConvergedDefaultSetUIRNorm()` can be used to
   base it on the initial residual norm instead. Unlike `KSPSolve()`, `KSPMatSolve()` does not remove the (transpose) null space of the operator from the block of
   right-hand sides, so the caller must make a singular system consistent by projecting `B` itself

   As in `KSPSolve()`, the iteration is delegated to the preconditioner when it provides a fast Richardson code, the convergence test is the default one or is skipped,
   and `Pmat` has no null space. `PCMatApplyRichardson()` is used when the preconditioner provides it, otherwise `PCApplyRichardson()` is applied
   to one right-hand side at a time so that `KSPMatSolve()` computes the same solutions as `KSPSolve()`. Since `KSPMatSolve()` is a stripped-down version of `KSPSolve()`,
   monitors are not called during a block iteration and so, unlike in `KSPSolve()`, they do not prevent this delegation

.seealso: [](ch_ksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`,
          `KSPRichardsonSetScale()`, `KSPPREONLY`, `KSPRichardsonSetSelfScale()`, `KSPMatSolve()`, `PCApplyRichardson()`, `PCMatApplyRichardson()`
M*/

PETSC_EXTERN PetscErrorCode KSPCreate_Richardson(KSP ksp)
{
  KSP_Richardson *richardsonP;

  PetscFunctionBegin;
  PetscCall(PetscNew(&richardsonP));
  ksp->data = (void *)richardsonP;

  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_PRECONDITIONED, PC_LEFT, 3));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_LEFT, 2));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_LEFT, 1));

  ksp->ops->setup          = KSPSetUp_Richardson;
  ksp->ops->solve          = KSPSolve_Richardson;
  ksp->ops->matsolve       = KSPMatSolve_Richardson;
  ksp->ops->destroy        = KSPDestroy_Richardson;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidual_Richardson;
  ksp->ops->view           = KSPView_Richardson;
  ksp->ops->setfromoptions = KSPSetFromOptions_Richardson;

  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPRichardsonSetScale_C", KSPRichardsonSetScale_Richardson));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPRichardsonSetSelfScale_C", KSPRichardsonSetSelfScale_Richardson));

  richardsonP->scale = 1.0;
  PetscFunctionReturn(PETSC_SUCCESS);
}
