#include <petsc/private/kspimpl.h>

typedef struct KSP_CG_PIPE_PR_s KSP_CG_PIPE_PR;
struct KSP_CG_PIPE_PR_s {
  PetscBool   rc_w_q; /* flag to determine whether w_k should be recomputer with A r_k */
};

/*
     KSPSetUp_PIPEPRCG - Sets up the workspace needed by the PIPEPRCG method.

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
static PetscErrorCode KSPSetUp_PIPEPRCG(KSP ksp)
{
  PetscFunctionBegin;
  /* get work vectors needed by PIPEPRCG */
  PetscCall(KSPSetWorkVecs(ksp,9));

  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_PIPEPRCG(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_CG_PIPE_PR *prcg=(KSP_CG_PIPE_PR*)ksp->data;
  PetscBool      flag=PETSC_FALSE;

  PetscFunctionBegin;
  PetscCall(PetscOptionsHead(PetscOptionsObject,"KSP PIPEPRCG options"));
  PetscCall(PetscOptionsBool("-recompute_w","-recompute w_k with Ar_k? (default = True)","",prcg->rc_w_q,&prcg->rc_w_q,&flag));
  if (!flag) prcg->rc_w_q = PETSC_TRUE;
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

/*
 KSPSolve_PIPEPRCG - This routine actually applies the pipelined predict and recompute conjugate gradient method
*/
static PetscErrorCode  KSPSolve_PIPEPRCG(KSP ksp)
{
  PetscInt       i;
  KSP_CG_PIPE_PR *prcg=(KSP_CG_PIPE_PR*)ksp->data;
  PetscScalar    alpha = 0.0, beta = 0.0, nu = 0.0, nu_old = 0.0, mudelgam[3], *mu_p, *delta_p, *gamma_p;
  PetscReal      dp    = 0.0;
  Vec            X,B,R,RT,W,WT,P,S,ST,U,UT,PRTST[3];
  Mat            Amat,Pmat;
  PetscBool      diagonalscale,rc_w_q=prcg->rc_w_q;

  /* note that these are pointers to entries of muldelgam, different than nu */
  mu_p=&mudelgam[0];delta_p=&mudelgam[1];gamma_p=&mudelgam[2];

  PetscFunctionBegin;

  PetscCall(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheck(!diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  R  = ksp->work[0];
  RT = ksp->work[1];
  W  = ksp->work[2];
  WT = ksp->work[3];
  P  = ksp->work[4];
  S  = ksp->work[5];
  ST = ksp->work[6];
  U  = ksp->work[7];
  UT = ksp->work[8];

  PetscCall(PCGetOperators(ksp->pc,&Amat,&Pmat));

  /* initialize */
  ksp->its = 0;
  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp,Amat,X,R));  /*   r <- b - Ax  */
    PetscCall(VecAYPX(R,-1.0,B));
  } else {
    PetscCall(VecCopy(B,R));               /*   r <- b       */
  }

  PetscCall(KSP_PCApply(ksp,R,RT));        /*   rt <- Br     */
  PetscCall(KSP_MatMult(ksp,Amat,RT,W));   /*   w <- A rt    */
  PetscCall(KSP_PCApply(ksp,W,WT));        /*   wt <- B w    */

  PetscCall(VecCopy(RT,P));                /*   p <- rt      */
  PetscCall(VecCopy(W,S));                 /*   p <- rt      */
  PetscCall(VecCopy(WT,ST));               /*   p <- rt      */

  PetscCall(KSP_MatMult(ksp,Amat,ST,U));   /*   u <- Ast     */
  PetscCall(KSP_PCApply(ksp,U,UT));        /*   ut <- Bu     */

  PetscCall(VecDotBegin(RT,R,&nu));
  PetscCall(VecDotBegin(P,S,mu_p));
  PetscCall(VecDotBegin(ST,S,gamma_p));

  PetscCall(VecDotEnd(RT,R,&nu));          /*   nu    <- (rt,r)  */
  PetscCall(VecDotEnd(P,S,mu_p));          /*   mu    <- (p,s)   */
  PetscCall(VecDotEnd(ST,S,gamma_p));      /*   gamma <- (st,s)  */
  *delta_p = *mu_p;

  i = 0;
  do {
    /* Compute appropriate norm */
    switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      PetscCall(VecNormBegin(RT,NORM_2,&dp));
      PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)RT)));
      PetscCall(VecNormEnd(RT,NORM_2,&dp));
      break;
    case KSP_NORM_UNPRECONDITIONED:
      PetscCall(VecNormBegin(R,NORM_2,&dp));
      PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));
      PetscCall(VecNormEnd(R,NORM_2,&dp));
      break;
    case KSP_NORM_NATURAL:
      dp = PetscSqrtReal(PetscAbsScalar(nu));
      break;
    case KSP_NORM_NONE:
      dp   = 0.0;
      break;
    default: SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
    }

    ksp->rnorm = dp;
    PetscCall(KSPLogResidualHistory(ksp,dp));
    PetscCall(KSPMonitor(ksp,i,dp));
    PetscCall((*ksp->converged)(ksp,i,dp,&ksp->reason,ksp->cnvP));
    if (ksp->reason) PetscFunctionReturn(0);

    /* update scalars */
    alpha = nu / *mu_p;
    nu_old = nu;
    nu = nu_old - 2.*alpha*(*delta_p) + (alpha*alpha)*(*gamma_p);
    beta = nu/nu_old;

    /* update vectors */
    PetscCall(VecAXPY(X, alpha,P));         /*   x  <- x  + alpha * p   */
    PetscCall(VecAXPY(R,-alpha,S));         /*   r  <- r  - alpha * s   */
    PetscCall(VecAXPY(RT,-alpha,ST));       /*   rt <- rt - alpha * st  */
    PetscCall(VecAXPY(W,-alpha,U));         /*   w  <- w  - alpha * u   */
    PetscCall(VecAXPY(WT,-alpha,UT));       /*   wt <- wt - alpha * ut  */
    PetscCall(VecAYPX(P,beta,RT));          /*   p  <- rt + beta  * p   */
    PetscCall(VecAYPX(S,beta,W));           /*   s  <- w  + beta  * s   */
    PetscCall(VecAYPX(ST,beta,WT));         /*   st <- wt + beta  * st  */

    PetscCall(VecDotBegin(RT,R,&nu));

    PRTST[0] = P; PRTST[1] = RT; PRTST[2] = ST;

    PetscCall(VecMDotBegin(S,3,PRTST,mudelgam));

    PetscCall(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));

    PetscCall(KSP_MatMult(ksp,Amat,ST,U));  /*   u  <- A st             */
    PetscCall(KSP_PCApply(ksp,U,UT));       /*   ut <- B u              */

    /* predict-and-recompute */
    /* ideally this is combined with the previous matvec; i.e. equivalent of MDot */
    if (rc_w_q) {
      PetscCall(KSP_MatMult(ksp,Amat,RT,W));  /*   w  <- A rt             */
      PetscCall(KSP_PCApply(ksp,W,WT));       /*   wt <- B w              */
    }

    PetscCall(VecDotEnd(RT,R,&nu));
    PetscCall(VecMDotEnd(S,3,PRTST,mudelgam));

    i++;
    ksp->its = i;

  } while (i<=ksp->max_it);
  if (!ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
   KSPPIPEPRCG - Pipelined predict-and-recompute conjugate gradient method.

   This method has only a single non-blocking reduction per iteration, compared to 2 blocking for standard CG.
   The non-blocking reduction is overlapped by the matrix-vector product and preconditioner application.

   Level: intermediate

   Notes:
   MPI configuration may be necessary for reductions to make asynchronous progress, which is important for performance of pipelined methods.
   See the FAQ on the PETSc website for details.

   Contributed by:
   Tyler Chen, University of Washington, Applied Mathematics Department

   Reference:
   Tyler Chen and Erin Carson. "Predict-and-recompute conjugate gradient variants." SIAM Journal on Scientific Computing 42.5 (2020): A3084-A3108.

   Acknowledgments:
   This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE-1762114. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author and do not necessarily reflect the views of the National Science Foundation.

.seealso: KSPCreate(), KSPSetType(), KSPPIPECG, KSPPIPECR, KSPGROPPCG, KSPPGMRES, KSPCG, KSPCGUseSingleReduction()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_PIPEPRCG(KSP ksp)
{
  KSP_CG_PIPE_PR *prcg=NULL;
  PetscBool      cite=PETSC_FALSE;

  PetscFunctionBegin;

  PetscCall(PetscCitationsRegister("@article{predict_and_recompute_cg,\n  author = {Tyler Chen and Erin C. Carson},\n  title = {Predict-and-recompute conjugate gradient variants},\n  journal = {},\n  year = {2020},\n  eprint = {1905.01549},\n  archivePrefix = {arXiv},\n  primaryClass = {cs.NA}\n}",&cite));

  PetscCall(PetscNewLog(ksp,&prcg));
  ksp->data = (void*)prcg;

  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));

  ksp->ops->setup          = KSPSetUp_PIPEPRCG;
  ksp->ops->solve          = KSPSolve_PIPEPRCG;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->view           = NULL;
  ksp->ops->setfromoptions = KSPSetFromOptions_PIPEPRCG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
