#ifndef lint
static char vcid[] = "$Id: cg.c,v 1.39 1996/03/21 02:42:11 curfman Exp curfman $";
#endif

/*                       
    By default, this code implements the CG (Conjugate Gradient) method,
    which is valid for real symmetric (and complex Hermitian) positive
    definite matrices. Note that for the complex Hermitian case, the
    VecDot() arguments within the code MUST remain in the order given
    for correct computation of inner products.

    Reference: Hestenes and Steifel, 1952.

    By switching to the indefinite vector inner product, VecTDot(), the
    same code is used for the complex symmetric case as well.  The user
    must call KSPCGSetType(ksp,CG_SYMMETRIC) or use the option 
    -ksp_cg_symmetric to invoke this variant for the complex case.
*/
#include <stdio.h>
#include <math.h>
#include "cgctx.h"

int KSPSetUp_CG(KSP ksp)
{
  KSP_CG *cgP = (KSP_CG *) ksp->data;
  int    maxit = ksp->max_it,ierr;

  /* check user parameters and functions */
  if (ksp->pc_side == PC_RIGHT)
    {SETERRQ(2,"KSPSetUp_CG:no right preconditioning for KSPCG");}
  else if (ksp->pc_side == PC_SYMMETRIC)
    {SETERRQ(2,"KSPSetUp_CG:no symmetric preconditioning for KSPCG");}
  if ((ierr = KSPCheckDef(ksp))) return ierr;

  /* get work vectors from user code */
  if ((ierr = KSPiDefaultGetWork( ksp, 3 ))) return ierr;

  if (ksp->calc_eigs) {
    /* get space to store tridiagonal matrix for Lanczo */
    cgP->e = (Scalar *) PetscMalloc(4*(maxit+1)*sizeof(Scalar)); CHKPTRQ(cgP->e);
    PLogObjectMemory(ksp,4*(maxit+1)*sizeof(Scalar));
    cgP->d  = cgP->e + maxit + 1; 
    cgP->ee = cgP->d + maxit + 1;
    cgP->dd = cgP->ee + maxit + 1;
  }
  return 0;
}

int  KSPSolve_CG(KSP ksp,int *its)
{
  int          ierr, i = 0,maxit,eigs,pres, hist_len, cerr;
  Scalar       dpi, a = 1.0,beta,betaold = 1.0,b,*e = 0,*d = 0, mone = -1.0, ma; 
  double       *history, dp;
  Vec          X,B,Z,R,P;
  KSP_CG       *cg;
  Mat          Amat, Pmat;
  MatStructure pflag;

  cg = (KSP_CG *) ksp->data;
  eigs    = ksp->calc_eigs;
  pres    = ksp->use_pres;
  maxit   = ksp->max_it;
  history = ksp->residual_history;
  hist_len= ksp->res_hist_size;
  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R       = ksp->work[0];
  Z       = ksp->work[1];
  P       = ksp->work[2];

#if !defined(PETSC_COMPLEX)
#define VecXDot(x,y,a) {ierr = VecDot(x,y,a); CHKERRQ(ierr)}
#else
#define VecXDot(x,y,a) \
  {if (cg->type == CG_HERMITIAN) {ierr = VecDot(x,y,a); CHKERRQ(ierr)} \
   else                          {ierr = VecTDot(x,y,a); CHKERRQ(ierr)}}
#endif

  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; b = 0.0; }
  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);

  if (!ksp->guess_zero) {
    ierr = MatMult(Amat,X,R); CHKERRQ(ierr);       /*   r <- b - Ax       */
    ierr = VecAYPX(&mone,B,R); CHKERRQ(ierr);
  }
  else { 
    ierr = VecCopy(B,R); CHKERRQ(ierr);            /*     r <- b (x is 0) */
  }
  ierr = PCApply(ksp->B,R,Z); CHKERRQ(ierr);       /*     z <- Br         */
  if (pres) {
      ierr = VecNorm(Z,NORM_2,&dp); CHKERRQ(ierr); /*    dp <- z'*z       */
  }
  else {
      ierr = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr); /*    dp <- r'*r       */
  }
  cerr = (*ksp->converged)(ksp,0,dp,ksp->cnvP);
  if (cerr) {*its =  0; return 0;}
  MONITOR(ksp,dp,0);
  if (history) history[0] = dp;

  for ( i=0; i<maxit; i++) {
     VecXDot(Z,R,&beta);                           /*     beta <- r'z     */
     if (i == 0) {
       if (beta == 0.0) break;
       ierr = VecCopy(Z,P); CHKERRQ(ierr);         /*     p <- z          */
     }
     else {
         b = beta/betaold;
#if !defined(PETSC_COMPLEX)
         if (b<0.0) SETERRQ(1,"KSPSolve_CG:Nonsymmetric/bad preconditioner");
#endif
         if (eigs) {
           e[i] = sqrt(b)/a;  
         }
         ierr = VecAYPX(&b,Z,P); CHKERRQ(ierr);    /*     p <- z + b* p   */
     }
     betaold = beta;
     ierr = MatMult(Amat,P,Z); CHKERRQ(ierr);      /*     z <- Kp         */
     VecXDot(P,Z,&dpi);                            /*     dpi <- z'p      */
     a = beta/dpi;                                 /*     a = beta/p'z    */
     if (eigs) {
       d[i] = sqrt(b)*e[i] + 1.0/a;
     }
     ierr = VecAXPY(&a,P,X); CHKERRQ(ierr);        /*     x <- x + ap     */
     ma = -a; VecAXPY(&ma,Z,R);                    /*     r <- r - az     */
     if (pres) {
       ierr = PCApply(ksp->B,R,Z); CHKERRQ(ierr);    /*     z <- Br         */
       ierr = VecNorm(Z,NORM_2,&dp); CHKERRQ(ierr);  /*    dp <- z'*z       */
     }
     else {
       ierr = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr);  /*    dp <- r'*r       */
     }
     if (history && hist_len > i + 1) history[i+1] = dp;
     MONITOR(ksp,dp,i+1);
     cerr = (*ksp->converged)(ksp,i+1,dp,ksp->cnvP);
     if (cerr) break;
     if (!pres) 
      {ierr = PCApply(ksp->B,R,Z); CHKERRQ(ierr);} /*     z <- Br         */
  }
  if (i == maxit) i--;
  if (history) ksp->res_act_size = (hist_len < i + 1) ? hist_len : i + 1;
  if (cerr <= 0) *its = -(i+1);
  else           *its = i+1;
  return 0;
}

int KSPDestroy_CG(PetscObject obj)
{
  KSP    ksp = (KSP) obj;
  KSP_CG *cg = (KSP_CG *) ksp->data;

  /* free space used for eigenvalue calculations */
  if ( ksp->calc_eigs ) {
    PetscFree(cg->e);
  }

  KSPiDefaultFreeWork( ksp );
  
  /* free the context variables */
  PetscFree(cg); 
  return 0;
}

/*@
    KSPCGSetType - Sets the variant of the conjugate gradient method to
    use for solving a linear system with a complex coefficient matrix.
    This option is irrelevant when solving a real system.

    Input Parameters:
.   ksp - the iterative context
.   type - the variant of CG to use, one of
$     CG_HERMITIAN - complex, Hermitian matrix (default)
$     CG_SYMMETRIC - complex, symmetric matrix

    Options Database Keys:
$   -ksp_cg_Hermitian
$   -ksp_cg_symmetric

    Note:
    By default, the matrix is assumed to be complex, Hermitian.

.keywords: CG, conjugate gradient, Hermitian, symmetric, set, type
@*/
int KSPCGSetType(KSP ksp,CGType type)
{
  KSP_CG *cg;
  PetscValidHeaderSpecific(ksp,KSP_COOKIE);
  cg = (KSP_CG *)ksp->data;
  if (ksp->type != KSPCG) return 0;
  cg->type = type;
  return 0;
}

static int KSPView_CG(PetscObject obj,Viewer viewer)
{
#if defined(PETSC_COMPLEX)
  KSP         ksp = (KSP)obj;
  KSP_CG      *cg = (KSP_CG *)ksp->data; 
  FILE        *fd;
  char        *cstr;
  int         ierr;
  ViewerType  vtype;

  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (cg->type == CG_HERMITIAN)
      PetscFPrintf(ksp->comm,fd,"    CG: variant for complex, Hermitian system\n");
    else if (cg->type == CG_SYMMETRIC)
      PetscFPrintf(ksp->comm,fd,"    CG: variant for complex, symmetric system\n");
    else
      PetscFPrintf(ksp->comm,fd,"    CG: unknown variant\n");
  }
#endif
  return 0;
}

int KSPCreate_CG(KSP ksp)
{
  KSP_CG *cg = (KSP_CG*) PetscMalloc(sizeof(KSP_CG));  CHKPTRQ(cg);
  PetscMemzero(cg,sizeof(KSP_CG));
  PLogObjectMemory(ksp,sizeof(KSP_CG));
#if !defined(PETSC_COMPLEX)
  cg->type                  = CG_SYMMETRIC;
#else
  cg->type                  = CG_HERMITIAN;
#endif
  ksp->data                 = (void *) cg;
  ksp->type                 = KSPCG;
  ksp->pc_side              = PC_LEFT;
  ksp->calc_res             = 1;
  ksp->setup                = KSPSetUp_CG;
  ksp->solver               = KSPSolve_CG;
  ksp->adjustwork           = KSPiDefaultAdjustWork;
  ksp->destroy              = KSPDestroy_CG;
  ksp->view                 = KSPView_CG;
  ksp->converged            = KSPDefaultConverged;
  ksp->buildsolution        = KSPDefaultBuildSolution;
  ksp->buildresidual        = KSPDefaultBuildResidual;
  return 0;
}




