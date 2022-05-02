
#include <petsc/private/kspimpl.h>

static PetscErrorCode KSPSetUp_BiCG(KSP ksp)
{
  PetscFunctionBegin;
  /* check user parameters and functions */
  PetscCheck(ksp->pc_side != PC_RIGHT,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"no right preconditioning for KSPBiCG");
  else PetscCheck(ksp->pc_side != PC_SYMMETRIC,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"no symmetric preconditioning for KSPBiCG");
  PetscCall(KSPSetWorkVecs(ksp,6));
  PetscFunctionReturn(0);
}

static PetscErrorCode  KSPSolve_BiCG(KSP ksp)
{
  PetscInt       i;
  PetscBool      diagonalscale;
  PetscScalar    dpi,a=1.0,beta,betaold=1.0,b,ma;
  PetscReal      dp;
  Vec            X,B,Zl,Zr,Rl,Rr,Pl,Pr;
  Mat            Amat,Pmat;

  PetscFunctionBegin;
  PetscCall(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheck(!diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X  = ksp->vec_sol;
  B  = ksp->vec_rhs;
  Rl = ksp->work[0];
  Zl = ksp->work[1];
  Pl = ksp->work[2];
  Rr = ksp->work[3];
  Zr = ksp->work[4];
  Pr = ksp->work[5];

  PetscCall(PCGetOperators(ksp->pc,&Amat,&Pmat));

  if (!ksp->guess_zero) {
    PetscCall(KSP_MatMult(ksp,Amat,X,Rr));      /*   r <- b - Ax       */
    PetscCall(VecAYPX(Rr,-1.0,B));
  } else {
    PetscCall(VecCopy(B,Rr));           /*     r <- b (x is 0) */
  }
  PetscCall(VecCopy(Rr,Rl));
  PetscCall(KSP_PCApply(ksp,Rr,Zr));     /*     z <- Br         */
  PetscCall(KSP_PCApplyHermitianTranspose(ksp,Rl,Zl));
  if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
    PetscCall(VecNorm(Zr,NORM_2,&dp));  /*    dp <- z'*z       */
  } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
    PetscCall(VecNorm(Rr,NORM_2,&dp));  /*    dp <- r'*r       */
  } else dp = 0.0;

  KSPCheckNorm(ksp,dp);
  PetscCall(KSPMonitor(ksp,0,dp));
  PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
  ksp->its   = 0;
  ksp->rnorm = dp;
  PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
  PetscCall(KSPLogResidualHistory(ksp,dp));
  PetscCall((*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
    PetscCall(VecDot(Zr,Rl,&beta));       /*     beta <- r'z     */
    KSPCheckDot(ksp,beta);
    if (!i) {
      if (beta == 0.0) {
        ksp->reason = KSP_DIVERGED_BREAKDOWN_BICG;
        PetscFunctionReturn(0);
      }
      PetscCall(VecCopy(Zr,Pr));       /*     p <- z          */
      PetscCall(VecCopy(Zl,Pl));
    } else {
      b    = beta/betaold;
      PetscCall(VecAYPX(Pr,b,Zr));  /*     p <- z + b* p   */
      b    = PetscConj(b);
      PetscCall(VecAYPX(Pl,b,Zl));
    }
    betaold = beta;
    PetscCall(KSP_MatMult(ksp,Amat,Pr,Zr)); /*     z <- Kp         */
    PetscCall(KSP_MatMultHermitianTranspose(ksp,Amat,Pl,Zl));
    PetscCall(VecDot(Zr,Pl,&dpi));            /*     dpi <- z'p      */
    KSPCheckDot(ksp,dpi);
    a       = beta/dpi;                           /*     a = beta/p'z    */
    PetscCall(VecAXPY(X,a,Pr));    /*     x <- x + ap     */
    ma      = -a;
    PetscCall(VecAXPY(Rr,ma,Zr));
    ma      = PetscConj(ma);
    PetscCall(VecAXPY(Rl,ma,Zl));
    if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
      PetscCall(KSP_PCApply(ksp,Rr,Zr));  /*     z <- Br         */
      PetscCall(KSP_PCApplyHermitianTranspose(ksp,Rl,Zl));
      PetscCall(VecNorm(Zr,NORM_2,&dp));  /*    dp <- z'*z       */
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      PetscCall(VecNorm(Rr,NORM_2,&dp));  /*    dp <- r'*r       */
    } else dp = 0.0;

    KSPCheckNorm(ksp,dp);
    PetscCall(PetscObjectSAWsTakeAccess((PetscObject)ksp));
    ksp->its   = i+1;
    ksp->rnorm = dp;
    PetscCall(PetscObjectSAWsGrantAccess((PetscObject)ksp));
    PetscCall(KSPLogResidualHistory(ksp,dp));
    PetscCall(KSPMonitor(ksp,i+1,dp));
    PetscCall((*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP));
    if (ksp->reason) break;
    if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      PetscCall(KSP_PCApply(ksp,Rr,Zr));  /* z <- Br  */
      PetscCall(KSP_PCApplyHermitianTranspose(ksp,Rl,Zl));
    }
    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*MC
     KSPBICG - Implements the Biconjugate gradient method (similar to running the conjugate
         gradient on the normal equations).

   Options Database Keys:
   see KSPSolve()

   Level: beginner

   Notes:
    this method requires that one be apply to apply the transpose of the preconditioner and operator
         as well as the operator and preconditioner.
         Supports only left preconditioning

         See KSPCGNE for code that EXACTLY runs the preconditioned conjugate gradient method on the
         normal equations

.seealso: `KSPCreate()`, `KSPSetType()`, `KSPType`, `KSP`, `KSPBCGS`, `KSPCGNE`

M*/
PETSC_EXTERN PetscErrorCode KSPCreate_BiCG(KSP ksp)
{
  PetscFunctionBegin;
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2));
  PetscCall(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));

  ksp->ops->setup          = KSPSetUp_BiCG;
  ksp->ops->solve          = KSPSolve_BiCG;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->view           = NULL;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
