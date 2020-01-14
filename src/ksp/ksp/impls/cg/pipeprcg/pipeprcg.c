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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* get work vectors needed by PIPEPRCG */
  ierr = KSPSetWorkVecs(ksp,9);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSetFromOptions_PIPEPRCG(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscInt       ierr=0;
  KSP_CG_PIPE_PR *prcg=(KSP_CG_PIPE_PR*)ksp->data;
  PetscBool      flag=PETSC_FALSE;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP PIPEPRCG options");CHKERRQ(ierr);
  PetscOptionsBool("-recompute_w","-recompute w_k with Ar_k? (default = True)","",prcg->rc_w_q,&prcg->rc_w_q,&flag);CHKERRQ(ierr);
  if (!flag) prcg->rc_w_q = PETSC_TRUE;
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
 KSPSolve_PIPEPRCG - This routine actually applies the pipelined predict and recompute conjugate gradient method

 Input Parameter:
 .     ksp - the Krylov space object that was set to use conjugate gradient, by, for
             example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPCG);
*/
static PetscErrorCode  KSPSolve_PIPEPRCG(KSP ksp)
{
  PetscErrorCode ierr;
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

  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

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

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  /* initialize */
  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);  /*   r <- b - Ax  */
    ierr = VecAYPX(R,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,R);CHKERRQ(ierr);               /*   r <- b       */
  }

  ierr = KSP_PCApply(ksp,R,RT);CHKERRQ(ierr);        /*   rt <- Br     */
  ierr = KSP_MatMult(ksp,Amat,RT,W);CHKERRQ(ierr);   /*   w <- A rt    */
  ierr = KSP_PCApply(ksp,W,WT);CHKERRQ(ierr);        /*   wt <- B w    */

  ierr = VecCopy(RT,P);CHKERRQ(ierr);                /*   p <- rt      */
  ierr = VecCopy(W,S);CHKERRQ(ierr);                 /*   p <- rt      */
  ierr = VecCopy(WT,ST);CHKERRQ(ierr);               /*   p <- rt      */

  ierr = KSP_MatMult(ksp,Amat,ST,U);CHKERRQ(ierr);   /*   u <- Ast     */
  ierr = KSP_PCApply(ksp,U,UT);CHKERRQ(ierr);        /*   ut <- Bu     */

  ierr = VecDotBegin(RT,R,&nu);CHKERRQ(ierr);
  ierr = VecDotBegin(P,S,mu_p);CHKERRQ(ierr);
  ierr = VecDotBegin(ST,S,gamma_p);CHKERRQ(ierr);

  ierr = VecDotEnd(RT,R,&nu);CHKERRQ(ierr);          /*   nu    <- (rt,r)  */
  ierr = VecDotEnd(P,S,mu_p);CHKERRQ(ierr);          /*   mu    <- (p,s)   */
  ierr = VecDotEnd(ST,S,gamma_p);CHKERRQ(ierr);      /*   gamma <- (st,s)  */
  *delta_p = *mu_p;

  i = 0;
  do {

   /* Compute appropriate norm */
   switch (ksp->normtype) {
     case KSP_NORM_PRECONDITIONED:
        ierr = VecNormBegin(RT,NORM_2,&dp);CHKERRQ(ierr);
        ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)RT));CHKERRQ(ierr);
        ierr = VecNormEnd(RT,NORM_2,&dp);CHKERRQ(ierr);
        break;
    case KSP_NORM_UNPRECONDITIONED:
        ierr = VecNormBegin(R,NORM_2,&dp);CHKERRQ(ierr);
        ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R));CHKERRQ(ierr);
        ierr = VecNormEnd(R,NORM_2,&dp);CHKERRQ(ierr);
        break;
    case KSP_NORM_NATURAL:
        dp = PetscSqrtReal(PetscAbsScalar(nu));
        break;
    case KSP_NORM_NONE:
        dp   = 0.0;
        break;
    default: SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"%s",KSPNormTypes[ksp->normtype]);
    }

    ksp->rnorm = dp;
    ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    ierr = KSPMonitor(ksp,i,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;

    /* update scalars */
    alpha = nu / *mu_p;
    nu_old = nu;
    nu = nu_old - 2*alpha*(*delta_p) + (alpha*alpha)*(*gamma_p);
    beta = nu/nu_old;

    /* update vectors */
    ierr = VecAXPY(X, alpha,P);CHKERRQ(ierr);         /*   x  <- x  + alpha * p   */
    ierr = VecAXPY(R,-alpha,S);CHKERRQ(ierr);         /*   r  <- r  - alpha * s   */
    ierr = VecAXPY(RT,-alpha,ST);CHKERRQ(ierr);       /*   rt <- rt - alpha * st  */
    ierr = VecAXPY(W,-alpha,U);CHKERRQ(ierr);         /*   w  <- w  - alpha * u   */
    ierr = VecAXPY(WT,-alpha,UT);CHKERRQ(ierr);       /*   wt <- wt - alpha * ut  */
    ierr = VecAYPX(P,beta,RT);CHKERRQ(ierr);          /*   p  <- rt + beta  * p   */
    ierr = VecAYPX(S,beta,W);CHKERRQ(ierr);           /*   s  <- w  + beta  * s   */
    ierr = VecAYPX(ST,beta,WT);CHKERRQ(ierr);         /*   st <- wt + beta  * st  */

    ierr = VecDotBegin(RT,R,&nu);CHKERRQ(ierr);

    PRTST[0] = P; PRTST[1] = RT; PRTST[2] = ST;

    ierr = VecMDotBegin(S,3,PRTST,mudelgam);CHKERRQ(ierr);

    ierr = PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)R));CHKERRQ(ierr);

    ierr = KSP_MatMult(ksp,Amat,ST,U);CHKERRQ(ierr);  /*   u  <- A st             */
    ierr = KSP_PCApply(ksp,U,UT);CHKERRQ(ierr);       /*   ut <- B u              */

    /* predict-and-recompute */
    /* ideally this is combined with the previous matvec; i.e. equivalent of MDot */
    if ( rc_w_q ) {
        ierr = KSP_MatMult(ksp,Amat,RT,W);CHKERRQ(ierr);  /*   w  <- A rt             */
        ierr = KSP_PCApply(ksp,W,WT);CHKERRQ(ierr);       /*   wt <- B w              */
    }

    ierr = VecDotEnd(RT,R,&nu);CHKERRQ(ierr);
    ierr = VecMDotEnd(S,3,PRTST,mudelgam);CHKERRQ(ierr);

    i++;
    ksp->its = i;

  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
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
   "Predict-and-recompute conjugate gradient variants". Tyler Chen and Erin C. Carson. In preparation.

   Acknowledgments:
   This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE-1762114. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author and do not necessarily reflect the views of the National Science Foundation.

.seealso: KSPCreate(), KSPSetType(), KSPPIPECG, KSPPIPECR, KSPGROPPCG, KSPPGMRES, KSPCG, KSPCGUseSingleReduction()
M*/
PETSC_EXTERN PetscErrorCode KSPCreate_PIPEPRCG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CG_PIPE_PR *prcg=NULL;
  PetscBool      cite=PETSC_FALSE;

  PetscFunctionBegin;

  ierr = PetscCitationsRegister("@article{predict_and_recompute_cg,\n  author = {Tyler Chen and Erin C. Carson},\n  title = {Predict-and-recompute conjugate gradient variants},\n  journal = {},\n  year = {2020},\n  eprint = {1905.01549},\n  archivePrefix = {arXiv},\n  primaryClass = {cs.NA}\n}",&cite);CHKERRQ(ierr);

  ierr = PetscNewLog(ksp,&prcg);CHKERRQ(ierr);
  ksp->data = (void*)prcg;

  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);

  ksp->ops->setup          = KSPSetUp_PIPEPRCG;
  ksp->ops->solve          = KSPSolve_PIPEPRCG;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->view           = 0;
  ksp->ops->setfromoptions = KSPSetFromOptions_PIPEPRCG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}

