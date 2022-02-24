
#include <../src/ksp/ksp/impls/lcd/lcdimpl.h>

PetscErrorCode KSPSetUp_LCD(KSP ksp)
{
  KSP_LCD        *lcd = (KSP_LCD*)ksp->data;
  PetscInt       restart = lcd->restart;

  PetscFunctionBegin;
  /* get work vectors needed by LCD */
  CHKERRQ(KSPSetWorkVecs(ksp,2));

  CHKERRQ(VecDuplicateVecs(ksp->work[0],restart+1,&lcd->P));
  CHKERRQ(VecDuplicateVecs(ksp->work[0], restart + 1, &lcd->Q));
  CHKERRQ(PetscLogObjectMemory((PetscObject)ksp,2*(restart+2)*sizeof(Vec)));
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
  PetscInt       it,j,max_k;
  PetscScalar    alfa, beta, num, den, mone;
  PetscReal      rnorm = 0.0;
  Vec            X,B,R,Z;
  KSP_LCD        *lcd;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  CHKERRQ(PCGetDiagonalScale(ksp->pc,&diagonalscale));
  PetscCheckFalse(diagonalscale,PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  lcd   = (KSP_LCD*)ksp->data;
  X     = ksp->vec_sol;
  B     = ksp->vec_rhs;
  R     = ksp->work[0];
  Z     = ksp->work[1];
  max_k = lcd->restart;
  mone  = -1;

  CHKERRQ(PCGetOperators(ksp->pc,&Amat,&Pmat));

  ksp->its = 0;
  if (!ksp->guess_zero) {
    CHKERRQ(KSP_MatMult(ksp,Amat,X,Z));             /*   z <- b - Ax       */
    CHKERRQ(VecAYPX(Z,mone,B));
  } else {
    CHKERRQ(VecCopy(B,Z));                         /*     z <- b (x is 0) */
  }

  CHKERRQ(KSP_PCApply(ksp,Z,R));                   /*     r <- M^-1z         */
  if (ksp->normtype != KSP_NORM_NONE) {
    CHKERRQ(VecNorm(R,NORM_2,&rnorm));
    KSPCheckNorm(ksp,rnorm);
  }
  CHKERRQ(KSPLogResidualHistory(ksp,rnorm));
  CHKERRQ(KSPMonitor(ksp,0,rnorm));
  ksp->rnorm = rnorm;

  /* test for convergence */
  CHKERRQ((*ksp->converged)(ksp,0,rnorm,&ksp->reason,ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);

  VecCopy(R,lcd->P[0]);

  while (!ksp->reason && ksp->its < ksp->max_it) {
    it   = 0;
    CHKERRQ(KSP_MatMult(ksp,Amat,lcd->P[it],Z));
    CHKERRQ(KSP_PCApply(ksp,Z,lcd->Q[it]));

    while (!ksp->reason && it < max_k && ksp->its < ksp->max_it) {
      ksp->its++;
      CHKERRQ(VecDot(lcd->P[it],R,&num));
      CHKERRQ(VecDot(lcd->P[it],lcd->Q[it], &den));
      KSPCheckDot(ksp,den);
      alfa = num/den;
      CHKERRQ(VecAXPY(X,alfa,lcd->P[it]));
      CHKERRQ(VecAXPY(R,-alfa,lcd->Q[it]));
      if (ksp->normtype != KSP_NORM_NONE) {
        CHKERRQ(VecNorm(R,NORM_2,&rnorm));
        KSPCheckNorm(ksp,rnorm);
      }

      ksp->rnorm = rnorm;
      CHKERRQ(KSPLogResidualHistory(ksp,rnorm));
      CHKERRQ(KSPMonitor(ksp,ksp->its,rnorm));
      CHKERRQ((*ksp->converged)(ksp,ksp->its,rnorm,&ksp->reason,ksp->cnvP));

      if (ksp->reason) break;

      CHKERRQ(VecCopy(R,lcd->P[it+1]));
      CHKERRQ(KSP_MatMult(ksp,Amat,lcd->P[it+1],Z));
      CHKERRQ(KSP_PCApply(ksp,Z,lcd->Q[it+1]));

      for (j = 0; j <= it; j++) {
        CHKERRQ(VecDot(lcd->P[j],lcd->Q[it+1],&num));
        KSPCheckDot(ksp,num);
        CHKERRQ(VecDot(lcd->P[j],lcd->Q[j],&den));
        beta = -num/den;
        CHKERRQ(VecAXPY(lcd->P[it+1],beta,lcd->P[j]));
        CHKERRQ(VecAXPY(lcd->Q[it+1],beta,lcd->Q[j]));
      }
      it++;
    }
    CHKERRQ(VecCopy(lcd->P[it],lcd->P[0]));
  }
  if (ksp->its >= ksp->max_it && !ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
  CHKERRQ(VecCopy(X,ksp->vec_sol));
  PetscFunctionReturn(0);
}
/*
       KSPDestroy_LCD - Frees all memory space used by the Krylov method

*/
PetscErrorCode KSPReset_LCD(KSP ksp)
{
  KSP_LCD        *lcd = (KSP_LCD*)ksp->data;

  PetscFunctionBegin;
  if (lcd->P) CHKERRQ(VecDestroyVecs(lcd->restart+1,&lcd->P));
  if (lcd->Q) CHKERRQ(VecDestroyVecs(lcd->restart+1,&lcd->Q));
  PetscFunctionReturn(0);
}

PetscErrorCode KSPDestroy_LCD(KSP ksp)
{
  PetscFunctionBegin;
  CHKERRQ(KSPReset_LCD(ksp));
  CHKERRQ(PetscFree(ksp->data));
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
  PetscBool      iascii;

  PetscFunctionBegin;
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  restart=%d\n",lcd->restart));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"  happy breakdown tolerance %g\n",lcd->haptol));
  }
  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_LCD - Checks the options database for options related to the
                            LCD method.
*/
PetscErrorCode KSPSetFromOptions_LCD(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscBool      flg;
  KSP_LCD        *lcd = (KSP_LCD*)ksp->data;

  PetscFunctionBegin;
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"KSP LCD options"));
  CHKERRQ(PetscOptionsInt("-ksp_lcd_restart","Number of vectors conjugate","KSPLCDSetRestart",lcd->restart,&lcd->restart,&flg));
  PetscCheckFalse(flg && lcd->restart < 1,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Restart must be positive");
  CHKERRQ(PetscOptionsReal("-ksp_lcd_haptol","Tolerance for exact convergence (happy ending)","KSPLCDSetHapTol",lcd->haptol,&lcd->haptol,&flg));
  PetscCheckFalse(flg && lcd->haptol < 0.0,PetscObjectComm((PetscObject)ksp),PETSC_ERR_ARG_OUTOFRANGE,"Tolerance must be non-negative");
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
+   * - J.Y. Yuan, G.H.Golub, R.J. Plemmons, and W.A.G. Cecilio. Semiconjugate
     direction methods for real positive definite system. BIT Numerical
     Mathematics, 44(1),2004.
.   * - Y. Dai and J.Y. Yuan. Study on semiconjugate direction methods for
     nonsymmetric systems. International Journal for Numerical Methods in
     Engineering, 60, 2004.
.   * - L. Catabriga, A.L.G.A. Coutinho, and L.P.Franca. Evaluating the LCD
     algorithm for solving linear systems of equations arising from implicit
     SUPG formulation of compressible flows. International Journal for
     Numerical Methods in Engineering, 60, 2004
-   * - L. Catabriga, A. M. P. Valli, B. Z. Melotti, L. M. Pessoa,
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
  KSP_LCD        *lcd;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(ksp,&lcd));
  ksp->data    = (void*)lcd;
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3));
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
