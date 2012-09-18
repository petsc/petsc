
#include <petsc-private/kspimpl.h>

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_BiCG"
PetscErrorCode KSPSetUp_BiCG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* check user parameters and functions */
  if (ksp->pc_side == PC_RIGHT) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"no right preconditioning for KSPBiCG");
  else if (ksp->pc_side == PC_SYMMETRIC) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"no symmetric preconditioning for KSPBiCG");
  ierr = KSPDefaultGetWork(ksp,6);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_BiCG"
PetscErrorCode  KSPSolve_BiCG(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i;
  PetscBool      diagonalscale;
  PetscScalar    dpi,a=1.0,beta,betaold=1.0,b,ma;
  PetscReal      dp;
  Vec            X,B,Zl,Zr,Rl,Rr,Pl,Pr;
  Mat            Amat,Pmat;
  MatStructure   pflag;

  PetscFunctionBegin;
  ierr    = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  Rl      = ksp->work[0];
  Zl      = ksp->work[1];
  Pl      = ksp->work[2];
  Rr      = ksp->work[3];
  Zr      = ksp->work[4];
  Pr      = ksp->work[5];

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,Rr);CHKERRQ(ierr);      /*   r <- b - Ax       */
    ierr = VecAYPX(Rr,-1.0,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,Rr);CHKERRQ(ierr);           /*     r <- b (x is 0) */
  }
  ierr = VecCopy(Rr,Rl);CHKERRQ(ierr);
  ierr = KSP_PCApply(ksp,Rr,Zr);CHKERRQ(ierr);     /*     z <- Br         */
  ierr = VecConjugate(Rl);CHKERRQ(ierr);
  ierr = KSP_PCApplyTranspose(ksp,Rl,Zl);CHKERRQ(ierr);
  ierr = VecConjugate(Rl);CHKERRQ(ierr);
  ierr = VecConjugate(Zl);CHKERRQ(ierr);
  if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
    ierr = VecNorm(Zr,NORM_2,&dp);CHKERRQ(ierr);  /*    dp <- z'*z       */
  } else {
    ierr = VecNorm(Rr,NORM_2,&dp);CHKERRQ(ierr);  /*    dp <- r'*r       */
  }
  ierr = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
  ksp->its   = 0;
  ksp->rnorm = dp;
  ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,dp);
  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
     ierr = VecDot(Zr,Rl,&beta);CHKERRQ(ierr);       /*     beta <- r'z     */
     if (!i) {
       if (beta == 0.0) {
         ksp->reason = KSP_DIVERGED_BREAKDOWN_BICG;
         PetscFunctionReturn(0);
       }
       ierr = VecCopy(Zr,Pr);CHKERRQ(ierr);       /*     p <- z          */
       ierr = VecCopy(Zl,Pl);CHKERRQ(ierr);
     } else {
       b = beta/betaold;
       ierr = VecAYPX(Pr,b,Zr);CHKERRQ(ierr);  /*     p <- z + b* p   */
       b = PetscConj(b);
       ierr = VecAYPX(Pl,b,Zl);CHKERRQ(ierr);
     }
     betaold = beta;
     ierr = KSP_MatMult(ksp,Amat,Pr,Zr);CHKERRQ(ierr);    /*     z <- Kp         */
     ierr = VecConjugate(Pl);CHKERRQ(ierr);
     ierr = KSP_MatMultTranspose(ksp,Amat,Pl,Zl);CHKERRQ(ierr);
     ierr = VecConjugate(Pl);CHKERRQ(ierr);
     ierr = VecConjugate(Zl);CHKERRQ(ierr);
     ierr = VecDot(Zr,Pl,&dpi);CHKERRQ(ierr);               /*     dpi <- z'p      */
     a = beta/dpi;                                 /*     a = beta/p'z    */
     ierr = VecAXPY(X,a,Pr);CHKERRQ(ierr);       /*     x <- x + ap     */
     ma = -a;
     ierr = VecAXPY(Rr,ma,Zr);CHKERRQ(ierr);
     ma = PetscConj(ma);
     ierr = VecAXPY(Rl,ma,Zl);CHKERRQ(ierr);
     if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
       ierr = KSP_PCApply(ksp,Rr,Zr);CHKERRQ(ierr);  /*     z <- Br         */
       ierr = VecConjugate(Rl);CHKERRQ(ierr);
       ierr = KSP_PCApplyTranspose(ksp,Rl,Zl);CHKERRQ(ierr);
       ierr = VecConjugate(Rl);CHKERRQ(ierr);
       ierr = VecConjugate(Zl);CHKERRQ(ierr);
       ierr = VecNorm(Zr,NORM_2,&dp);CHKERRQ(ierr);  /*    dp <- z'*z       */
     } else {
       ierr = VecNorm(Rr,NORM_2,&dp);CHKERRQ(ierr);  /*    dp <- r'*r       */
     }
     ierr = PetscObjectTakeAccess(ksp);CHKERRQ(ierr);
     ksp->its   = i+1;
     ksp->rnorm = dp;
     ierr = PetscObjectGrantAccess(ksp);CHKERRQ(ierr);
     KSPLogResidualHistory(ksp,dp);
     ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
     ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
     if (ksp->reason) break;
     if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
       ierr = KSP_PCApply(ksp,Rr,Zr);CHKERRQ(ierr);  /* z <- Br  */
       ierr = VecConjugate(Rl);CHKERRQ(ierr);
       ierr = KSP_PCApplyTranspose(ksp,Rl,Zl);CHKERRQ(ierr);
       ierr = VecConjugate(Rl);CHKERRQ(ierr);
       ierr = VecConjugate(Zl);CHKERRQ(ierr);
     }
     i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

/*MC
     KSPBICG - Implements the Biconjugate gradient method (similar to running the conjugate
         gradient on the normal equations).

   Options Database Keys:
.   see KSPSolve()

   Level: beginner

   Notes: this method requires that one be apply to apply the transpose of the preconditioner and operator
         as well as the operator and preconditioner.
         Supports only left preconditioning

         See KSPCGNE for code that EXACTLY runs the preconditioned conjugate gradient method on the
         normal equations

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP, KSPBCGS, KSPCGNE

M*/
EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_BiCG"
PetscErrorCode  KSPCreate_BiCG(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ksp->data                      = (void*)0;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,1);CHKERRQ(ierr);

  ksp->ops->setup                = KSPSetUp_BiCG;
  ksp->ops->solve                = KSPSolve_BiCG;
  ksp->ops->destroy              = KSPDefaultDestroy;
  ksp->ops->view                 = 0;
  ksp->ops->setfromoptions       = 0;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;

  PetscFunctionReturn(0);
}
EXTERN_C_END





