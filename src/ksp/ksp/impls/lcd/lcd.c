
#include <../src/ksp/ksp/impls/lcd/lcdimpl.h>

PetscErrorCode KSPSetUp_LCD(KSP ksp)
{
  KSP_LCD        *lcd = (KSP_LCD*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       restart = lcd->restart;

  PetscFunctionBegin;
  /* get work vectors needed by LCD */
  ierr = KSPSetWorkVecs(ksp,2);CHKERRQ(ierr);

  ierr = VecDuplicateVecs(ksp->work[0],restart+1,&lcd->P);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ksp->work[0], restart + 1, &lcd->Q);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory((PetscObject)ksp,2*(restart+2)*sizeof(Vec));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*     KSPSolve_LCD - This routine actually applies the left conjugate
    direction method

   Input Parameter:
.     ksp - the Krylov space object that was set to use LCD, by, for
            example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPLCD);

   Output Parameter:
.     its - number of iterations used

*/
PetscErrorCode  KSPSolve_LCD(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       it,j,max_k;
  PetscScalar    alfa, beta, num, den, mone;
  PetscReal      rnorm = 0.0;
  Vec            X,B,R,Z;
  KSP_LCD        *lcd;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  PetscAssertFalse(diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  lcd   = (KSP_LCD*)ksp->data;
  X     = ksp->vec_sol;
  B     = ksp->vec_rhs;
  R     = ksp->work[0];
  Z     = ksp->work[1];
  max_k = lcd->restart;
  mone  = -1;

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,Z);CHKERRQ(ierr);             /*   z <- b - Ax       */
    ierr = VecAYPX(Z,mone,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,Z);CHKERRQ(ierr);                         /*     z <- b (x is 0) */
  }

  ierr = KSP_PCApply(ksp,Z,R);CHKERRQ(ierr);                   /*     r <- M^-1z         */
  if (ksp->normtype != KSP_NORM_NONE) {
    ierr = VecNorm(R,NORM_2,&rnorm);CHKERRQ(ierr);
    KSPCheckNorm(ksp,rnorm);
  }
  ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,rnorm);CHKERRQ(ierr);
  ksp->rnorm = rnorm;

  /* test for convergence */
  ierr = (*ksp->converged)(ksp,0,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  VecCopy(R,lcd->P[0]);

  while (!ksp->reason && ksp->its < ksp->max_it) {
    it   = 0;
    ierr = KSP_MatMult(ksp,Amat,lcd->P[it],Z);CHKERRQ(ierr);
    ierr = KSP_PCApply(ksp,Z,lcd->Q[it]);CHKERRQ(ierr);

    while (!ksp->reason && it < max_k && ksp->its < ksp->max_it) {
      ksp->its++;
      ierr = VecDot(lcd->P[it],R,&num);CHKERRQ(ierr);
      ierr = VecDot(lcd->P[it],lcd->Q[it], &den);CHKERRQ(ierr);
      KSPCheckDot(ksp,den);
      alfa = num/den;
      ierr = VecAXPY(X,alfa,lcd->P[it]);CHKERRQ(ierr);
      ierr = VecAXPY(R,-alfa,lcd->Q[it]);CHKERRQ(ierr);
      if (ksp->normtype != KSP_NORM_NONE) {
        ierr = VecNorm(R,NORM_2,&rnorm);CHKERRQ(ierr);
        KSPCheckNorm(ksp,rnorm);
      }

      ksp->rnorm = rnorm;
      ierr = KSPLogResidualHistory(ksp,rnorm);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,ksp->its,rnorm);CHKERRQ(ierr);
      ierr = (*ksp->converged)(ksp,ksp->its,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);

      if (ksp->reason) break;

      ierr = VecCopy(R,lcd->P[it+1]);CHKERRQ(ierr);
      ierr = KSP_MatMult(ksp,Amat,lcd->P[it+1],Z);CHKERRQ(ierr);
      ierr = KSP_PCApply(ksp,Z,lcd->Q[it+1]);CHKERRQ(ierr);

      for (j = 0; j <= it; j++) {
        ierr = VecDot(lcd->P[j],lcd->Q[it+1],&num);CHKERRQ(ierr);
        KSPCheckDot(ksp,num);
        ierr = VecDot(lcd->P[j],lcd->Q[j],&den);CHKERRQ(ierr);
        beta = -num/den;
        ierr = VecAXPY(lcd->P[it+1],beta,lcd->P[j]);CHKERRQ(ierr);
        ierr = VecAXPY(lcd->Q[it+1],beta,lcd->Q[j]);CHKERRQ(ierr);
      }
      it++;
    }
    ierr = VecCopy(lcd->P[it],lcd->P[0]);CHKERRQ(ierr);
  }
  if (ksp->its >= ksp->max_it && !ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
  ierr = VecCopy(X,ksp->vec_sol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
/*
       KSPDestroy_LCD - Frees all memory space used by the Krylov method

*/
PetscErrorCode KSPReset_LCD(KSP ksp)
{
  KSP_LCD        *lcd = (KSP_LCD*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (lcd->P) { ierr = VecDestroyVecs(lcd->restart+1,&lcd->P);CHKERRQ(ierr);}
  if (lcd->Q) { ierr = VecDestroyVecs(lcd->restart+1,&lcd->Q);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_LCD(KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPReset_LCD(ksp);CHKERRQ(ierr);
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     KSPView_LCD - Prints information about the current Krylov method being used

      Currently this only prints information to a file (or stdout) about the
      symmetry of the problem. If your Krylov method has special options or
      flags that information should be printed here.

*/
PetscErrorCode KSPView_LCD(KSP ksp,PetscViewer viewer)
{

  KSP_LCD        *lcd = (KSP_LCD*)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  restart=%d\n",lcd->restart);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  happy breakdown tolerance %g\n",lcd->haptol);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_LCD - Checks the options database for options related to the
                            LCD method.
*/
PetscErrorCode KSPSetFromOptions_LCD(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  KSP_LCD        *lcd = (KSP_LCD*)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP LCD options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_lcd_restart","Number of vectors conjugate","KSPLCDSetRestart",lcd->restart,&lcd->restart,&flg);CHKERRQ(ierr);
  PetscAssertFalse(flg && lcd->restart < 1,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Restart must be positive");
  ierr = PetscOptionsReal("-ksp_lcd_haptol","Tolerance for exact convergence (happy ending)","KSPLCDSetHapTol",lcd->haptol,&lcd->haptol,&flg);CHKERRQ(ierr);
  PetscAssertFalse(flg && lcd->haptol < 0.0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Tolerance must be non-negative");
  PetscFunctionReturn(0);
}

/*MC
     KSPLCD -  Implements the LCD (left conjugate direction) method in PETSc.

   Options Database Keys:
+   -ksp_lcd_restart - number of vectors conjudate
-   -ksp_lcd_haptol - tolerance for exact convergence (happing ending)

   Level: beginner

    Notes:
    Support only for left preconditioning

    References:
+    1. - J.Y. Yuan, G.H.Golub, R.J. Plemmons, and W.A.G. Cecilio. Semiconjugate
     direction methods for real positive definite system. BIT Numerical
     Mathematics, 44(1),2004.
.    2. - Y. Dai and J.Y. Yuan. Study on semiconjugate direction methods for
     nonsymmetric systems. International Journal for Numerical Methods in
     Engineering, 60, 2004.
.    3. - L. Catabriga, A.L.G.A. Coutinho, and L.P.Franca. Evaluating the LCD
     algorithm for solving linear systems of equations arising from implicit
     SUPG formulation of compressible flows. International Journal for
     Numerical Methods in Engineering, 60, 2004
-    4. - L. Catabriga, A. M. P. Valli, B. Z. Melotti, L. M. Pessoa,
     A. L. G. A. Coutinho, Performance of LCD iterative method in the finite
     element and finite difference solution of convection diffusion
     equations,  Communications in Numerical Methods in Engineering, (Early
     View).

  Contributed by: Lucia Catabriga <luciac@ices.utexas.edu>

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPCGSetType(), KSPLCDSetRestart(), KSPLCDSetHapTol()

M*/

PETSC_EXTERN PetscErrorCode KSPCreate_LCD(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_LCD        *lcd;

  PetscFunctionBegin;
  ierr         = PetscNewLog(ksp,&lcd);CHKERRQ(ierr);
  ksp->data    = (void*)lcd;
  ierr         = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);
  ierr         = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  lcd->restart = 30;
  lcd->haptol  = 1.0e-30;

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup          = KSPSetUp_LCD;
  ksp->ops->solve          = KSPSolve_LCD;
  ksp->ops->reset          = KSPReset_LCD;
  ksp->ops->destroy        = KSPDestroy_LCD;
  ksp->ops->view           = KSPView_LCD;
  ksp->ops->setfromoptions = KSPSetFromOptions_LCD;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  PetscFunctionReturn(0);
}
