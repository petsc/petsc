
#include <../src/ksp/ksp/impls/lcd/lcdimpl.h>
#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_LCD"

PetscErrorCode KSPSetUp_LCD(KSP ksp)
{
  KSP_LCD         *lcd = (KSP_LCD*)ksp->data;
  PetscErrorCode ierr;
  PetscInt        restart = lcd->restart;

  PetscFunctionBegin;
  /* get work vectors needed by LCD */
  ierr = KSPDefaultGetWork(ksp,2);CHKERRQ(ierr);

  ierr = VecDuplicateVecs(ksp->work[0],restart+1,&lcd->P);CHKERRQ(ierr);
  ierr = VecDuplicateVecs(ksp->work[0], restart + 1, &lcd->Q);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,2*(restart+2)*sizeof(Vec));CHKERRQ(ierr);
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
#undef __FUNCT__
#define __FUNCT__ "KSPSolve_LCD"
PetscErrorCode  KSPSolve_LCD(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       it,j,max_k;
  PetscScalar    alfa, beta, num, den, mone;
  PetscReal      rnorm;
  Vec            X,B,R,Z;
  KSP_LCD        *lcd;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  PetscBool      diagonalscale;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);

  lcd            = (KSP_LCD*)ksp->data;
  X              = ksp->vec_sol;
  B              = ksp->vec_rhs;
  R              = ksp->work[0];
  Z              = ksp->work[1];
  max_k          = lcd->restart;
  mone = -1;

  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,Z);CHKERRQ(ierr);             /*   z <- b - Ax       */
    ierr = VecAYPX(Z,mone,B);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(B,Z);CHKERRQ(ierr);                         /*     z <- b (x is 0) */
  }

  ierr = KSP_PCApply(ksp,Z,R);CHKERRQ(ierr);                   /*     r <- M^-1z         */
  ierr = VecNorm(R,NORM_2,&rnorm);CHKERRQ(ierr);
  KSPLogResidualHistory(ksp,rnorm);
  ierr = KSPMonitor(ksp,0,rnorm);CHKERRQ(ierr);
  ksp->rnorm = rnorm;

   /* test for convergence */
  ierr = (*ksp->converged)(ksp,0,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);

  it = 0;
  VecCopy(R,lcd->P[0]);

  while (!ksp->reason && ksp->its < ksp->max_it) {
    it = 0;
    ierr = KSP_MatMult(ksp,Amat,lcd->P[it],Z);CHKERRQ(ierr);
    ierr = KSP_PCApply(ksp,Z,lcd->Q[it]);CHKERRQ(ierr);

    while(!ksp->reason && it < max_k && ksp->its < ksp->max_it) {
      ksp->its++;
      ierr = VecDot(lcd->P[it],R,&num);CHKERRQ(ierr);
      ierr = VecDot(lcd->P[it],lcd->Q[it], &den);CHKERRQ(ierr);
      alfa = num/den;
      ierr = VecAXPY(X,alfa,lcd->P[it]);CHKERRQ(ierr);
      ierr = VecAXPY(R,-alfa,lcd->Q[it]);CHKERRQ(ierr);
      ierr = VecNorm(R,NORM_2,&rnorm);CHKERRQ(ierr);

      ksp->rnorm = rnorm;
      KSPLogResidualHistory(ksp,rnorm);
      ierr = KSPMonitor(ksp,ksp->its,rnorm);CHKERRQ(ierr);
      ierr = (*ksp->converged)(ksp,ksp->its,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);

      if (ksp->reason) break;

      ierr = VecCopy(R,lcd->P[it+1]);CHKERRQ(ierr);
      ierr = KSP_MatMult(ksp,Amat,lcd->P[it+1],Z);CHKERRQ(ierr);
      ierr = KSP_PCApply(ksp,Z,lcd->Q[it+1]);CHKERRQ(ierr);

      for ( j = 0; j <= it; j++)	{
        ierr = VecDot(lcd->P[j],lcd->Q[it+1],&num);CHKERRQ(ierr);
        ierr = VecDot(lcd->P[j],lcd->Q[j],&den);CHKERRQ(ierr);
        beta = - num/den;
        ierr = VecAXPY(lcd->P[it+1],beta,lcd->P[j]);CHKERRQ(ierr);
        ierr = VecAXPY(lcd->Q[it+1],beta,lcd->Q[j]);CHKERRQ(ierr);
      }
      it++;
    }
    ierr = VecCopy(lcd->P[it],lcd->P[0]);CHKERRQ(ierr);
  }
  if (ksp->its >= ksp->max_it && !ksp->reason) ksp->reason = KSP_DIVERGED_ITS;
  ierr = VecCopy(X,ksp->vec_sol);

  PetscFunctionReturn(0);
}
/*
       KSPDestroy_LCD - Frees all memory space used by the Krylov method

*/
#undef __FUNCT__
#define __FUNCT__ "KSPReset_LCD"
PetscErrorCode KSPReset_LCD(KSP ksp)
{
  KSP_LCD         *lcd = (KSP_LCD*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (lcd->P) { ierr = VecDestroyVecs(lcd->restart+1,&lcd->P);CHKERRQ(ierr);}
  if (lcd->Q) { ierr = VecDestroyVecs(lcd->restart+1,&lcd->Q);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}


#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_LCD"
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
#undef __FUNCT__
#define __FUNCT__ "KSPView_LCD"
PetscErrorCode KSPView_LCD(KSP ksp,PetscViewer viewer)
{

  KSP_LCD         *lcd = (KSP_LCD *)ksp->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
      ierr = PetscViewerASCIIPrintf(viewer,"  LCD: restart=%d\n",lcd->restart);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"  LCD: happy breakdown tolerance %g\n",lcd->haptol);CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)ksp)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for KSP LCD",((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_LCD - Checks the options database for options related to the
                            LCD method.
*/
#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_LCD"
PetscErrorCode KSPSetFromOptions_LCD(KSP ksp)
{
  PetscErrorCode ierr;
  PetscBool      flg;
  KSP_LCD        *lcd = (KSP_LCD *)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("KSP LCD options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_lcd_restart","Number of vectors conjugate","KSPLCDSetRestart",lcd->restart,&lcd->restart,&flg);CHKERRQ(ierr);
  if (flg && lcd->restart < 1) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Restart must be positive");
  ierr = PetscOptionsReal("-ksp_lcd_haptol","Tolerance for exact convergence (happy ending)","KSPLCDSetHapTol",lcd->haptol,&lcd->haptol,&flg);CHKERRQ(ierr);
  if (flg && lcd->haptol < 0.0) SETERRQ(((PetscObject)ksp)->comm,PETSC_ERR_ARG_OUTOFRANGE,"Tolerance must be non-negative");
  PetscFunctionReturn(0);
}

/*MC
     KSPLCD -  Implements the LCD (left conjugate direction) method in PETSc.

   Options Database Keys:
+   -ksp_lcd_restart - number of vectors conjudate
-   -ksp_lcd_haptol - tolerance for exact convergence (happing ending)

   Level: beginner

    Notes: Support only for left preconditioning

    References:
   - J.Y. Yuan, G.H.Golub, R.J. Plemmons, and W.A.G. Cecilio. Semiconjugate
     direction methods for real positive definite system. BIT Numerical
     Mathematics, 44(1):189-207,2004.
   - Y. Dai and J.Y. Yuan. Study on semi-conjugate direction methods for
     non-symmetric systems. International Journal for Numerical Methods in
     Engineering, 60:1383-1399,2004.
   - L. Catabriga, A.L.G.A. Coutinho, and L.P.Franca. Evaluating the LCD
     algorithm for solving linear systems of equations arising from implicit
     SUPG formulation of compressible flows. International Journal for
     Numerical Methods in Engineering, 60:1513-1534,2004
   - L. Catabriga, A. M. P. Valli, B. Z. Melotti, L. M. Pessoa,
     A. L. G. A. Coutinho, Performance of LCD iterative method in the finite
     element and finite difference solution of convection-diffusion
     equations,  Communications in Numerical Methods in Engineering, (Early
     View).

  Contributed by: Lucia Catabriga <luciac@ices.utexas.edu>


.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPCGSetType(), KSPLCDSetRestart(), KSPLCDSetHapTol()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__
#define __FUNCT__ "KSPCreate_LCD"
PetscErrorCode KSPCreate_LCD(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_LCD         *lcd;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_LCD,&lcd);CHKERRQ(ierr);
  ksp->data                      = (void*)lcd;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  lcd->restart                   = 30;
  lcd->haptol                    = 1.0e-30;

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup                = KSPSetUp_LCD;
  ksp->ops->solve                = KSPSolve_LCD;
  ksp->ops->reset                = KSPReset_LCD;
  ksp->ops->destroy              = KSPDestroy_LCD;
  ksp->ops->view                 = KSPView_LCD;
  ksp->ops->setfromoptions       = KSPSetFromOptions_LCD;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;

  PetscFunctionReturn(0);
}
EXTERN_C_END





