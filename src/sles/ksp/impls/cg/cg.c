#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: cg.c,v 1.86 1999/04/19 22:14:43 bsmith Exp bsmith $";
#endif

/*
    This file implements the conjugate gradient method in PETSc as part of
    KSP. You can use this as a starting point for implementing your own 
    Krylov method that is not provided with PETSc.

    The following basic routines are required for each Krylov method.
        KSPCreate_XXX()          - Creates the Krylov context
        KSPSetFromOptions_XXX()  - Sets runtime options
        KSPSolve_XXX()           - Runs the Krylov method
        KSPDestroy_XXX(0         - Destroys the Krylov context, freeing all 
                                   memory it needed
    Here the "_XXX" denotes a particular implementation, in this case 
    we use _CG (e.g. KSPCreate_CG, KSPDestroy_CG). These routines are 
    are actually called vai the common user interface routines
    KSPSetType(), KSPSetFromOptions(), KSPSolve(), and KSPDestroy() so the
    application code interface remains identical for all preconditioners.

    Other basic routines for the KSP objects include
        KSPSetUp_XXX()
        KSPPrintHelp_XXX()        - Prints details of runtime options
        KSPView_XXX()             - Prints details of solver being used.

    Detailed notes:                         
    By default, this code implements the CG (Conjugate Gradient) method,
    which is valid for real symmetric (and complex Hermitian) positive
    definite matrices. Note that for the complex Hermitian case, the
    VecDot() arguments within the code MUST remain in the order given
    for correct computation of inner products.

    Reference: Hestenes and Steifel, 1952.

    By switching to the indefinite vector inner product, VecTDot(), the
    same code is used for the complex symmetric case as well.  The user
    must call KSPCGSetType(ksp,KSP_CG_SYMMETRIC) or use the option 
    -ksp_cg_symmetric to invoke this variant for the complex case.
    Note, however, that the complex symmetric code is NOT valid for
    all such matrices ... and thus we don't recommend using this method.
*/
/*
       cgctx.h defines the simple data structured used to store information
    related to the type of matrix (e.g. complex symmetric) being solved and
    data used during the optional Lanczo process used to compute eigenvalues
*/
#include "src/sles/ksp/impls/cg/cgctx.h"       /*I "ksp.h" I*/
extern int KSPComputeExtremeSingularValues_CG(KSP,double *,double *);
extern int KSPComputeEigenvalues_CG(KSP,int,double *,double *,int *);

/*
     KSPSetUp_CG - Sets up the workspace needed by the CG method. 

      This is called once, usually automatically by SLESSolve() or SLESSetUp()
     but can be called directly by KSPSetUp()
*/
#undef __FUNC__  
#define __FUNC__ "KSPSetUp_CG"
int KSPSetUp_CG(KSP ksp)
{
  KSP_CG *cgP = (KSP_CG *) ksp->data;
  int    maxit = ksp->max_it,ierr;

  PetscFunctionBegin;
  /* 
       This implementation of CG only handles left preconditioning
     so generate an error otherwise.
  */
  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(2,0,"No right preconditioning for KSPCG");
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(2,0,"No symmetric preconditioning for KSPCG");
  }

  /* get work vectors needed by CG */
  ierr = KSPDefaultGetWork( ksp, 3 ); CHKERRQ(ierr);

  /*
     If user requested computations of eigenvalues then allocate work
     work space needed
  */
  if (ksp->calc_sings) {
    /* get space to store tridiagonal matrix for Lanczos */
    cgP->e = (Scalar *) PetscMalloc(2*(maxit+1)*sizeof(Scalar));CHKPTRQ(cgP->e);
    PLogObjectMemory(ksp,2*(maxit+1)*sizeof(Scalar));
    cgP->d                         = cgP->e + maxit + 1; 
    cgP->ee = (double *)PetscMalloc(2*(maxit+1)*sizeof(double));CHKPTRQ(cgP->ee);
    PLogObjectMemory(ksp,2*(maxit+1)*sizeof(Scalar));
    cgP->dd                        = cgP->ee + maxit + 1;
    ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_CG;
    ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_CG;
  }
  PetscFunctionReturn(0);
}

/*
       KSPSolve_CG - This routine actually applies the conjugate gradient 
    method

   Input Parameter:
.     ksp - the Krylov space object that was set to use conjugate gradient, by, for 
            example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPCG);

   Output Parameter:
.     its - number of iterations used

*/
#undef __FUNC__  
#define __FUNC__ "KSPSolve_CG"
int  KSPSolve_CG(KSP ksp,int *its)
{
  int          ierr, i = 0,maxit,eigs,pres, cerr;
  Scalar       dpi, a = 1.0,beta,betaold = 1.0,b,*e = 0,*d = 0, mone = -1.0, ma; 
  double       dp = 0.0;
  Vec          X,B,Z,R,P;
  KSP_CG       *cg;
  Mat          Amat, Pmat;
  MatStructure pflag;

  PetscFunctionBegin;
  cg = (KSP_CG *) ksp->data;
  eigs    = ksp->calc_sings;
  pres    = ksp->use_pres;
  maxit   = ksp->max_it;
  X       = ksp->vec_sol;
  B       = ksp->vec_rhs;
  R       = ksp->work[0];
  Z       = ksp->work[1];
  P       = ksp->work[2];

#if !defined(USE_PETSC_COMPLEX)
#define VecXDot(x,y,a) {ierr = VecDot(x,y,a); CHKERRQ(ierr)}
#else
#define VecXDot(x,y,a) \
  {if (cg->type == KSP_CG_HERMITIAN) {ierr = VecDot(x,y,a); CHKERRQ(ierr)} \
   else                              {ierr = VecTDot(x,y,a); CHKERRQ(ierr)}}
#endif

  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; b = 0.0; }
  ierr = PCGetOperators(ksp->B,&Amat,&Pmat,&pflag); CHKERRQ(ierr);

  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = MatMult(Amat,X,R); CHKERRQ(ierr);         /*   r <- b - Ax       */
    ierr = VecAYPX(&mone,B,R); CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,R); CHKERRQ(ierr);              /*     r <- b (x is 0) */
  }
  ierr = PCApply(ksp->B,R,Z); CHKERRQ(ierr);         /*     z <- Br         */
  if (!ksp->avoidnorms) {
    if (pres) {
        ierr = VecNorm(Z,NORM_2,&dp); CHKERRQ(ierr); /*    dp <- z'*z       */
    } else {
        ierr = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr); /*    dp <- r'*r       */
    }
  }
  cerr = (*ksp->converged)(ksp,0,dp,ksp->cnvP);      /* test for convergence */
  if (cerr) {*its =  0; PetscFunctionReturn(0);}
  KSPLogResidualHistory(ksp,dp);
  KSPMonitor(ksp,0,dp);                              /* call any registered monitor routines */
  ksp->rnorm              = dp;

  for ( i=0; i<maxit; i++) {
     ksp->its = i+1;
     VecXDot(Z,R,&beta);                           /*     beta <- r'z     */
     if (i == 0) {
       if (beta == 0.0) break;
       ierr = VecCopy(Z,P); CHKERRQ(ierr);         /*     p <- z          */
     } else {
         b = beta/betaold;
#if !defined(USE_PETSC_COMPLEX)
         if (b < 0.0) SETERRQ( PETSC_ERR_KSP_BRKDWN,0,"Nonsymmetric/bad preconditioner");
#endif
         if (eigs) {
           e[i] = sqrt(PetscAbsScalar(b))/a;  
         }
         ierr = VecAYPX(&b,Z,P); CHKERRQ(ierr);    /*     p <- z + b* p   */
     }
     betaold = beta;
     ierr = MatMult(Amat,P,Z); CHKERRQ(ierr);      /*     z <- Kp         */
     VecXDot(P,Z,&dpi);                            /*     dpi <- z'p      */
     a = beta/dpi;                                 /*     a = beta/p'z    */
     if (eigs) {
       d[i] = sqrt(PetscAbsScalar(b))*e[i] + 1.0/a;
     }
     ierr = VecAXPY(&a,P,X); CHKERRQ(ierr);          /*     x <- x + ap     */
     ma = -a; VecAXPY(&ma,Z,R);                      /*     r <- r - az     */
     if (pres) {
       ierr = PCApply(ksp->B,R,Z); CHKERRQ(ierr);    /*     z <- Br         */
       if (!ksp->avoidnorms) {
         ierr = VecNorm(Z,NORM_2,&dp); CHKERRQ(ierr);/*    dp <- z'*z       */
       }
     }
     else if (!ksp->avoidnorms) {
       ierr = VecNorm(R,NORM_2,&dp); CHKERRQ(ierr);  /*    dp <- r'*r       */
     }
     ksp->rnorm = dp;
     KSPLogResidualHistory(ksp,dp);
     KSPMonitor(ksp,i+1,dp);
     cerr = (*ksp->converged)(ksp,i+1,dp,ksp->cnvP);
     if (cerr) break;
     if (!pres) {ierr = PCApply(ksp->B,R,Z); CHKERRQ(ierr);} /* z <- Br  */
  }
  if (i == maxit) {i--; ksp->its--;}
  if (cerr <= 0) *its = -(i+1);
  else           *its = i+1;
  PetscFunctionReturn(0);
}
/*
       KSPDestroy_CG - Frees all memory space used by the Krylov method

*/
#undef __FUNC__  
#define __FUNC__ "KSPDestroy_CG" 
int KSPDestroy_CG(KSP ksp)
{
  KSP_CG *cg = (KSP_CG *) ksp->data;
  int    ierr;

  PetscFunctionBegin;
  /* free space used for singular value calculations */
  if ( ksp->calc_sings ) {
    PetscFree(cg->e);
    PetscFree(cg->ee);
  }

  ierr = KSPDefaultFreeWork( ksp );CHKERRQ(ierr);
  
  /* free the context variable */
  PetscFree(cg); 
  PetscFunctionReturn(0);
}

/*
     KSPView_CG - Prints information about the current Krylov method being used

      Currently this only prints information to a file (or stdout) about the 
      symmetry of the problem. If your Krylov method has special options or 
      flags that information should be printed here.

*/
#undef __FUNC__  
#define __FUNC__ "KSPView_CG" 
int KSPView_CG(KSP ksp,Viewer viewer)
{
#if defined(USE_PETSC_COMPLEX)
  KSP_CG      *cg = (KSP_CG *)ksp->data; 
  int         ierr;
  ViewerType  vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (PetscTypeCompare(vtype,ASCII_VIEWER)) {
    if (cg->type == KSP_CG_HERMITIAN) {
      ierr = ViewerASCIIPrintf(viewer,"  CG: variant for complex, Hermitian system\n");CHKERRQ(ierr);
    } else if (cg->type == KSP_CG_SYMMETRIC) {
      ierr = ViewerASCIIPrintf(viewer,"  CG: variant for complex, symmetric system\n");CHKERRQ(ierr);
    } else {
      ierr = ViewerASCIIPrintf(viewer,"  CG: unknown variant\n");CHKERRQ(ierr);
    }
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
#endif
  PetscFunctionReturn(0);
}

/*
    KSPPrint_Help - Prints a help message that indicates what run time options are
                    available for this solver
*/
#undef __FUNC__  
#define __FUNC__ "KSPPrintHelp_CG"
static int KSPPrintHelp_CG(KSP ksp,char *p)
{
#if defined(USE_PETSC_COMPLEX)
  int ierr;
#endif

  PetscFunctionBegin;
#if defined(USE_PETSC_COMPLEX)
  ierr = (*PetscHelpPrintf)(ksp->comm," Options for CG method:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_cg_Hermitian: use CG for complex, Hermitian matrix (default)\n",p);CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(ksp->comm,"   %sksp_cg_symmetric: use CG for complex, symmetric matrix\n",p);CHKERRQ(ierr);
#endif

  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_CG - Checks the options database for options related to the 
                           conjugate gradient method. Any options checked here should
                           also be indicated with the KSPPrintHelp_XXX() routine.
*/ 
#undef __FUNC__  
#define __FUNC__ "KSPSetFromOptions_CG"
int KSPSetFromOptions_CG(KSP ksp)
{
  int       ierr,flg;

  PetscFunctionBegin;
  ierr = OptionsHasName(ksp->prefix,"-ksp_cg_Hermitian",&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPCGSetType(ksp,KSP_CG_HERMITIAN);CHKERRQ(ierr); }
  ierr = OptionsHasName(ksp->prefix,"-ksp_cg_symmetric",&flg);CHKERRQ(ierr);
  if (flg) { ierr = KSPCGSetType(ksp,KSP_CG_SYMMETRIC);CHKERRQ(ierr); }

  PetscFunctionReturn(0);
}

/*
    KSPCGSetType_CG - This is an option that is SPECIFIC to this particular Krylov method.
                      This routine is registered below in KSPCreate_CG() and called from the 
                      routine KSPCGSetType() (see the file cgtype.c).

        This must be wrapped in an EXTERN_C_BEGIN to be dynamically linkable in C++
*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPCGSetType_CG" 
int KSPCGSetType_CG(KSP ksp,KSPCGType type)
{
  KSP_CG *cg;

  PetscFunctionBegin;
  cg = (KSP_CG *)ksp->data;
  cg->type = type;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/*
    KSPCreate_CG - Creates the data structure for the Krylov method CG and sets the 
       function pointers for all the routines it needs to call (KSPSolve_CG() etc)

    It must be wrapped in EXTERN_C_BEGIN to be dynamically linkable in C++
*/
EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "KSPCreate_CG"
int KSPCreate_CG(KSP ksp)
{
  int    ierr;
  KSP_CG *cg = (KSP_CG*) PetscMalloc(sizeof(KSP_CG));  CHKPTRQ(cg);

  PetscFunctionBegin;
  PetscMemzero(cg,sizeof(KSP_CG));
  PLogObjectMemory(ksp,sizeof(KSP_CG));
#if !defined(USE_PETSC_COMPLEX)
  cg->type                       = KSP_CG_SYMMETRIC;
#else
  cg->type                       = KSP_CG_HERMITIAN;
#endif
  ksp->data                      = (void *) cg;
  ksp->pc_side                   = PC_LEFT;
  ksp->calc_res                  = 1;

  /*
       Sets the functions that are associated with this data structure 
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup                = KSPSetUp_CG;
  ksp->ops->solve                = KSPSolve_CG;
  ksp->ops->destroy              = KSPDestroy_CG;
  ksp->ops->view                 = KSPView_CG;
  ksp->ops->printhelp            = KSPPrintHelp_CG;
  ksp->ops->setfromoptions       = KSPSetFromOptions_CG;
  ksp->converged                 = KSPDefaultConverged;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;

  /*
      Attach the function KSPCGSetType_CG() to this object. The routine 
      KSPCGSetType() checks for this attached function and calls it if it finds
      it. (Sort of like a dynamic member function that can be added at run time
  */
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGSetType_C","KSPCGSetType_CG",
                                     (void*)KSPCGSetType_CG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END




