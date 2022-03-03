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
  CHKERRQ(KSPSetWorkVecs(ksp,9));

  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_PIPEPRCG(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  KSP_CG_PIPE_PR *prcg=(KSP_CG_PIPE_PR*)ksp->data;
  PetscBool      flag=PETSC_FALSE;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"KSP PIPEPRCG options"));
  CHKERRQ(PetscOptionsBool("-recompute_w","-recompute w_k with Ar_k? (default = True)","",prcg->rc_w_q,&prcg->rc_w_q,&flag));
  if (!flag) prcg->rc_w_q = PETSC_TRUE;
  CHKERRQ(PetscOptionsTail());
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

  CHKERRQ(PCGetDiagonalScale(ksp->pc,&diagonalscale));
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

  CHKERRQ(PCGetOperators(ksp->pc,&Amat,&Pmat));

  /* initialize */
  ksp->its = 0;
  if (!ksp->guess_zero) {
    CHKERRQ(KSP_MatMult(ksp,Amat,X,R));  /*   r <- b - Ax  */
    CHKERRQ(VecAYPX(R,-1.0,B));
  } else {
    CHKERRQ(VecCopy(B,R));               /*   r <- b       */
  }

  CHKERRQ(KSP_PCApply(ksp,R,RT));        /*   rt <- Br     */
  CHKERRQ(KSP_MatMult(ksp,Amat,RT,W));   /*   w <- A rt    */
  CHKERRQ(KSP_PCApply(ksp,W,WT));        /*   wt <- B w    */

  CHKERRQ(VecCopy(RT,P));                /*   p <- rt      */
  CHKERRQ(VecCopy(W,S));                 /*   p <- rt      */
  CHKERRQ(VecCopy(WT,ST));               /*   p <- rt      */

  CHKERRQ(KSP_MatMult(ksp,Amat,ST,U));   /*   u <- Ast     */
  CHKERRQ(KSP_PCApply(ksp,U,UT));        /*   ut <- Bu     */

  CHKERRQ(VecDotBegin(RT,R,&nu));
  CHKERRQ(VecDotBegin(P,S,mu_p));
  CHKERRQ(VecDotBegin(ST,S,gamma_p));

  CHKERRQ(VecDotEnd(RT,R,&nu));          /*   nu    <- (rt,r)  */
  CHKERRQ(VecDotEnd(P,S,mu_p));          /*   mu    <- (p,s)   */
  CHKERRQ(VecDotEnd(ST,S,gamma_p));      /*   gamma <- (st,s)  */
  *delta_p = *mu_p;

  i = 0;
  do {
    /* Compute appropriate norm */
    switch (ksp->normtype) {
    case KSP_NORM_PRECONDITIONED:
      CHKERRQ(VecNormBegin(RT,NORM_2,&dp));
      CHKERRQ(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)RT)));
      CHKERRQ(VecNormEnd(RT,NORM_2,&dp));
      break;
    case KSP_NORM_UNPRECONDITIONED:
      CHKERRQ(VecNormBegin(R,NORM_2,&dp));
      CHKERRQ(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));
      CHKERRQ(VecNormEnd(R,NORM_2,&dp));
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
    CHKERRQ(KSPLogResidualHistory(ksp,dp));
    CHKERRQ(KSPMonitor(ksp,i,dp));
    CHKERRQ((*ksp->converged)(ksp,i,dp,&ksp->reason,ksp->cnvP));
    if (ksp->reason) PetscFunctionReturn(0);

    /* update scalars */
    alpha = nu / *mu_p;
    nu_old = nu;
    nu = nu_old - 2.*alpha*(*delta_p) + (alpha*alpha)*(*gamma_p);
    beta = nu/nu_old;

    /* update vectors */
    CHKERRQ(VecAXPY(X, alpha,P));         /*   x  <- x  + alpha * p   */
    CHKERRQ(VecAXPY(R,-alpha,S));         /*   r  <- r  - alpha * s   */
    CHKERRQ(VecAXPY(RT,-alpha,ST));       /*   rt <- rt - alpha * st  */
    CHKERRQ(VecAXPY(W,-alpha,U));         /*   w  <- w  - alpha * u   */
    CHKERRQ(VecAXPY(WT,-alpha,UT));       /*   wt <- wt - alpha * ut  */
    CHKERRQ(VecAYPX(P,beta,RT));          /*   p  <- rt + beta  * p   */
    CHKERRQ(VecAYPX(S,beta,W));           /*   s  <- w  + beta  * s   */
    CHKERRQ(VecAYPX(ST,beta,WT));         /*   st <- wt + beta  * st  */

    CHKERRQ(VecDotBegin(RT,R,&nu));

    PRTST[0] = P; PRTST[1] = RT; PRTST[2] = ST;

    CHKERRQ(VecMDotBegin(S,3,PRTST,mudelgam));

    CHKERRQ(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R)));

    CHKERRQ(KSP_MatMult(ksp,Amat,ST,U));  /*   u  <- A st             */
    CHKERRQ(KSP_PCApply(ksp,U,UT));       /*   ut <- B u              */

    /* predict-and-recompute */
    /* ideally this is combined with the previous matvec; i.e. equivalent of MDot */
    if (rc_w_q) {
      CHKERRQ(KSP_MatMult(ksp,Amat,RT,W));  /*   w  <- A rt             */
      CHKERRQ(KSP_PCApply(ksp,W,WT));       /*   wt <- B w              */
    }

    CHKERRQ(VecDotEnd(RT,R,&nu));
    CHKERRQ(VecMDotEnd(S,3,PRTST,mudelgam));

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

  CHKERRQ(PetscCitationsRegister("@article{predict_and_recompute_cg,\n  author = {Tyler Chen and Erin C. Carson},\n  title = {Predict-and-recompute conjugate gradient variants},\n  journal = {},\n  year = {2020},\n  eprint = {1905.01549},\n  archivePrefix = {arXiv},\n  primaryClass = {cs.NA}\n}",&cite));

  CHKERRQ(PetscNewLog(ksp,&prcg));
  ksp->data = (void*)prcg;

  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));

  ksp->ops->setup          = KSPSetUp_PIPEPRCG;
  ksp->ops->solve          = KSPSolve_PIPEPRCG;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->view           = NULL;
  ksp->ops->setfromoptions = KSPSetFromOptions_PIPEPRCG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
