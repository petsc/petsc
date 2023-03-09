
#include <petsc/private/kspimpl.h> /*I "petscksp.h" I*/

typedef struct {
  PetscInt     restart;
  PetscInt     n_restarts;
  PetscScalar *val;
  Vec         *VV, *SS;
  Vec          R;

  PetscErrorCode (*modifypc)(KSP, PetscInt, PetscReal, void *); /* function to modify the preconditioner*/
  PetscErrorCode (*modifypc_destroy)(void *);                   /* function to destroy the user context for the modifypc function */

  void *modifypc_ctx; /* user defined data for the modifypc function */
} KSP_GCR;

static PetscErrorCode KSPSolve_GCR_cycle(KSP ksp)
{
  KSP_GCR    *ctx = (KSP_GCR *)ksp->data;
  PetscScalar r_dot_v;
  Mat         A, B;
  PC          pc;
  Vec         s, v, r;
  /*
     The residual norm will not be computed when ksp->its > ksp->chknorm hence need to initialize norm_r with some dummy value
  */
  PetscReal norm_r = 0.0, nrm;
  PetscInt  k, i, restart;
  Vec       x;

  PetscFunctionBegin;
  restart = ctx->restart;
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(KSPGetOperators(ksp, &A, &B));

  x = ksp->vec_sol;
  r = ctx->R;

  for (k = 0; k < restart; k++) {
    v = ctx->VV[k];
    s = ctx->SS[k];
    if (ctx->modifypc) PetscCall((*ctx->modifypc)(ksp, ksp->its, ksp->rnorm, ctx->modifypc_ctx));

    PetscCall(KSP_PCApply(ksp, r, s));    /* s = B^{-1} r */
    PetscCall(KSP_MatMult(ksp, A, s, v)); /* v = A s */

    PetscCall(VecMDot(v, k, ctx->VV, ctx->val));
    for (i = 0; i < k; i++) ctx->val[i] = -ctx->val[i];
    PetscCall(VecMAXPY(v, k, ctx->val, ctx->VV)); /* v = v - sum_{i=0}^{k-1} alpha_i v_i */
    PetscCall(VecMAXPY(s, k, ctx->val, ctx->SS)); /* s = s - sum_{i=0}^{k-1} alpha_i s_i */

    PetscCall(VecDotNorm2(r, v, &r_dot_v, &nrm));
    nrm     = PetscSqrtReal(nrm);
    r_dot_v = r_dot_v / nrm;
    PetscCall(VecScale(v, 1.0 / nrm));
    PetscCall(VecScale(s, 1.0 / nrm));
    PetscCall(VecAXPY(x, r_dot_v, s));
    PetscCall(VecAXPY(r, -r_dot_v, v));
    if (ksp->its > ksp->chknorm && ksp->normtype != KSP_NORM_NONE) {
      PetscCall(VecNorm(r, NORM_2, &norm_r));
      KSPCheckNorm(ksp, norm_r);
    }
    /* update the local counter and the global counter */
    ksp->its++;
    ksp->rnorm = norm_r;

    PetscCall(KSPLogResidualHistory(ksp, norm_r));
    PetscCall(KSPMonitor(ksp, ksp->its, norm_r));

    if (ksp->its - 1 > ksp->chknorm) {
      PetscCall((*ksp->converged)(ksp, ksp->its, norm_r, &ksp->reason, ksp->cnvP));
      if (ksp->reason) break;
    }

    if (ksp->its >= ksp->max_it) {
      ksp->reason = KSP_CONVERGED_ITS;
      break;
    }
  }
  ctx->n_restarts++;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSolve_GCR(KSP ksp)
{
  KSP_GCR  *ctx = (KSP_GCR *)ksp->data;
  Mat       A, B;
  Vec       r, b, x;
  PetscReal norm_r = 0.0;

  PetscFunctionBegin;
  PetscCall(KSPGetOperators(ksp, &A, &B));
  x = ksp->vec_sol;
  b = ksp->vec_rhs;
  r = ctx->R;

  /* compute initial residual */
  PetscCall(KSP_MatMult(ksp, A, x, r));
  PetscCall(VecAYPX(r, -1.0, b)); /* r = b - A x  */
  if (ksp->normtype != KSP_NORM_NONE) {
    PetscCall(VecNorm(r, NORM_2, &norm_r));
    KSPCheckNorm(ksp, norm_r);
  }
  ksp->its    = 0;
  ksp->rnorm0 = norm_r;

  PetscCall(KSPLogResidualHistory(ksp, ksp->rnorm0));
  PetscCall(KSPMonitor(ksp, ksp->its, ksp->rnorm0));
  PetscCall((*ksp->converged)(ksp, ksp->its, ksp->rnorm0, &ksp->reason, ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS);

  do {
    PetscCall(KSPSolve_GCR_cycle(ksp));
    if (ksp->reason) PetscFunctionReturn(PETSC_SUCCESS); /* catch case when convergence occurs inside the cycle */
  } while (ksp->its < ksp->max_it);

  if (ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPView_GCR(KSP ksp, PetscViewer viewer)
{
  KSP_GCR  *ctx = (KSP_GCR *)ksp->data;
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "  restart = %" PetscInt_FMT " \n", ctx->restart));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  restarts performed = %" PetscInt_FMT " \n", ctx->n_restarts));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetUp_GCR(KSP ksp)
{
  KSP_GCR  *ctx = (KSP_GCR *)ksp->data;
  Mat       A;
  PetscBool diagonalscale;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc, &diagonalscale));
  PetscCheck(!diagonalscale, PetscObjectComm((PetscObject)ksp), PETSC_ERR_SUP, "Krylov method %s does not support diagonal scaling", ((PetscObject)ksp)->type_name);

  PetscCall(KSPGetOperators(ksp, &A, NULL));
  PetscCall(MatCreateVecs(A, &ctx->R, NULL));
  PetscCall(VecDuplicateVecs(ctx->R, ctx->restart, &ctx->VV));
  PetscCall(VecDuplicateVecs(ctx->R, ctx->restart, &ctx->SS));

  PetscCall(PetscMalloc1(ctx->restart, &ctx->val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPReset_GCR(KSP ksp)
{
  KSP_GCR *ctx = (KSP_GCR *)ksp->data;

  PetscFunctionBegin;
  PetscCall(VecDestroy(&ctx->R));
  PetscCall(VecDestroyVecs(ctx->restart, &ctx->VV));
  PetscCall(VecDestroyVecs(ctx->restart, &ctx->SS));
  if (ctx->modifypc_destroy) PetscCall((*ctx->modifypc_destroy)(ctx->modifypc_ctx));
  PetscCall(PetscFree(ctx->val));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPDestroy_GCR(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPReset_GCR(ksp));
  PetscCall(KSPDestroyDefault(ksp));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGCRSetRestart_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGCRGetRestart_C", NULL));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGCRSetModifyPC_C", NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPSetFromOptions_GCR(KSP ksp, PetscOptionItems *PetscOptionsObject)
{
  KSP_GCR  *ctx = (KSP_GCR *)ksp->data;
  PetscInt  restart;
  PetscBool flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "KSP GCR options");
  PetscCall(PetscOptionsInt("-ksp_gcr_restart", "Number of Krylov search directions", "KSPGCRSetRestart", ctx->restart, &restart, &flg));
  if (flg) PetscCall(KSPGCRSetRestart(ksp, restart));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Force these parameters to not be EXTERN_C */
typedef PetscErrorCode (*KSPGCRModifyPCFunction)(KSP, PetscInt, PetscReal, void *);
typedef PetscErrorCode (*KSPGCRDestroyFunction)(void *);

static PetscErrorCode KSPGCRSetModifyPC_GCR(KSP ksp, KSPGCRModifyPCFunction function, void *data, KSPGCRDestroyFunction destroy)
{
  KSP_GCR *ctx = (KSP_GCR *)ksp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(ksp, KSP_CLASSID, 1);
  ctx->modifypc         = function;
  ctx->modifypc_destroy = destroy;
  ctx->modifypc_ctx     = data;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
 KSPGCRSetModifyPC - Sets the routine used by `KSPGCR` to modify the preconditioner for each iteration

 Logically Collective

 Input Parameters:
 +  ksp      - iterative context obtained from KSPCreate()
 .  function - user defined function to modify the preconditioner
 .  ctx      - user provided context for the modify preconditioner function
 -  destroy  - the function to use to destroy the user provided application context.

 Calling Sequence of `function`:
$  PetscErrorCode function (KSP ksp, PetscInt n, PetscReal rnorm, void *ctx)
+  ksp   - iterative context
.  n     - the total number of GCR iterations that have occurred
.  rnorm - 2-norm residual value
-  ctx   - the user provided application context

 Calling Sequence of `destroy`:
$ PetscErrorCode destroy(void *ctx)

 Level: intermediate

 Note:
 The default modifypc routine is `KSPGCRModifyPCNoChange()`

 Developer Note:
 The API should make uniform for all flexible types, [](sec_flexibleksp), and not have separate function calls for each type.

.seealso: [](chapter_ksp), `KSP`, `KSPGCR`, `KSPGCRModifyPCNoChange()`, [](sec_flexibleksp)
 @*/
PetscErrorCode KSPGCRSetModifyPC(KSP ksp, PetscErrorCode (*function)(KSP, PetscInt, PetscReal, void *), void *data, PetscErrorCode (*destroy)(void *))
{
  PetscFunctionBegin;
  PetscUseMethod(ksp, "KSPGCRSetModifyPC_C", (KSP, PetscErrorCode(*)(KSP, PetscInt, PetscReal, void *), void *data, PetscErrorCode (*)(void *)), (ksp, function, data, destroy));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPGCRSetRestart_GCR(KSP ksp, PetscInt restart)
{
  KSP_GCR *ctx;

  PetscFunctionBegin;
  ctx          = (KSP_GCR *)ksp->data;
  ctx->restart = restart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPGCRGetRestart_GCR(KSP ksp, PetscInt *restart)
{
  KSP_GCR *ctx;

  PetscFunctionBegin;
  ctx      = (KSP_GCR *)ksp->data;
  *restart = ctx->restart;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   KSPGCRSetRestart - Sets number of iterations at which `KSPGCR` restarts.

   Not Collective

   Input Parameters:
+  ksp - the Krylov space context
-  restart - integer restart value

   Options Database Key:
.   -ksp_gcr_restart <restart> - the number of stored vectors to orthogonalize against

   Level: intermediate

   Note:
   The default value is 30.

   Developer Note:
   The API could be made uniform for all `KSP` methods have have a restart.

.seealso: [](chapter_ksp), `KSPGCR`, `KSPSetTolerances()`, `KSPGCRGetRestart()`, `KSPGMRESSetRestart()`
@*/
PetscErrorCode KSPGCRSetRestart(KSP ksp, PetscInt restart)
{
  PetscFunctionBegin;
  PetscTryMethod(ksp, "KSPGCRSetRestart_C", (KSP, PetscInt), (ksp, restart));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
   KSPGCRGetRestart - Gets number of iterations at which `KSPGCR` restarts.

   Not Collective

   Input Parameter:
.  ksp - the Krylov space context

   Output Parameter:
.   restart - integer restart value

   Level: intermediate

.seealso: [](chapter_ksp), `KSPGCR`, `KSPSetTolerances()`, `KSPGCRSetRestart()`, `KSPGMRESGetRestart()`
@*/
PetscErrorCode KSPGCRGetRestart(KSP ksp, PetscInt *restart)
{
  PetscFunctionBegin;
  PetscTryMethod(ksp, "KSPGCRGetRestart_C", (KSP, PetscInt *), (ksp, restart));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPBuildSolution_GCR(KSP ksp, Vec v, Vec *V)
{
  Vec x;

  PetscFunctionBegin;
  x = ksp->vec_sol;
  if (v) {
    PetscCall(VecCopy(x, v));
    if (V) *V = v;
  } else if (V) {
    *V = ksp->vec_sol;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode KSPBuildResidual_GCR(KSP ksp, Vec t, Vec v, Vec *V)
{
  KSP_GCR *ctx;

  PetscFunctionBegin;
  ctx = (KSP_GCR *)ksp->data;
  if (v) {
    PetscCall(VecCopy(ctx->R, v));
    if (V) *V = v;
  } else if (V) {
    *V = ctx->R;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     KSPGCR - Implements the preconditioned flexible Generalized Conjugate Residual method. [](sec_flexibleksp),

   Options Database Key:
.   -ksp_gcr_restart <restart> - the number of stored vectors to orthogonalize against

   Level: beginner

    Notes:
    The GCR Krylov method supports non-symmetric matrices and permits the use of a preconditioner
    which may vary from one iteration to the next.

    Users can can define a method to vary the
           preconditioner between iterates via `KSPGCRSetModifyPC()`.

           Restarts are solves with x0 not equal to zero. When a restart occurs, the initial starting
           solution is given by the current estimate for x which was obtained by the last restart
           iterations of the GCR algorithm.

           Unlike `KSPGMRES` and `KSPFGMRES`, when using GCR, the solution and residual vector can be directly accessed at any iterate,
           with zero computational cost, via a call to `KSPBuildSolution()` and `KSPBuildResidual()` respectively.

           This implementation of GCR will only apply the stopping condition test whenever ksp->its > ksp->chknorm,
           where ksp->chknorm is specified via the command line argument -ksp_check_norm_iteration or via
           the function `KSPSetCheckNormIteration()`. Hence the residual norm reported by the monitor and stored
           in the residual history will be listed as 0.0 before this iteration. It is actually not 0.0; just not calculated.

           The method implemented requires the storage of 2 x restart + 1 vectors, twice as much as `KSPGMRES`.
           Support only for right preconditioning.

    Contributed by:
    Dave May

    References:
.   * - S. C. Eisenstat, H. C. Elman, and H. C. Schultz. Variational iterative methods for
           nonsymmetric systems of linear equations. SIAM J. Numer. Anal., 20, 1983

.seealso: [](chapter_ksp), [](sec_flexibleksp), `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPGCRSetRestart()`, `KSPGCRGetRestart()`,
          `KSPGCRSetRestart()`, `KSPGCRSetModifyPC()`, `KSPGMRES`, `KSPFGMRES`
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_GCR(KSP ksp)
{
  KSP_GCR *ctx;

  PetscFunctionBegin;
  PetscCall(PetscNew(&ctx));

  ctx->restart    = 30;
  ctx->n_restarts = 0;
  ksp->data       = (void *)ctx;

  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_NONE, PC_RIGHT, 1));
  PetscCall(KSPSetSupportedNorm(ksp, KSP_NORM_UNPRECONDITIONED, PC_RIGHT, 3));

  ksp->ops->setup          = KSPSetUp_GCR;
  ksp->ops->solve          = KSPSolve_GCR;
  ksp->ops->reset          = KSPReset_GCR;
  ksp->ops->destroy        = KSPDestroy_GCR;
  ksp->ops->view           = KSPView_GCR;
  ksp->ops->setfromoptions = KSPSetFromOptions_GCR;
  ksp->ops->buildsolution  = KSPBuildSolution_GCR;
  ksp->ops->buildresidual  = KSPBuildResidual_GCR;

  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGCRSetRestart_C", KSPGCRSetRestart_GCR));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGCRGetRestart_C", KSPGCRGetRestart_GCR));
  PetscCall(PetscObjectComposeFunction((PetscObject)ksp, "KSPGCRSetModifyPC_C", KSPGCRSetModifyPC_GCR));
  PetscFunctionReturn(PETSC_SUCCESS);
}
