
/*
    This file implements the conjugate gradient method in PETSc as part of
    KSP. You can use this as a starting point for implementing your own 
    Krylov method that is not provided with PETSc.

    The following basic routines are required for each Krylov method.
        KSPCreate_XXX()          - Creates the Krylov context
        KSPSetFromOptions_XXX()  - Sets runtime options
        KSPSolve_XXX()           - Runs the Krylov method
        KSPDestroy_XXX()         - Destroys the Krylov context, freeing all 
                                   memory it needed
    Here the "_XXX" denotes a particular implementation, in this case 
    we use _CG (e.g. KSPCreate_CG, KSPDestroy_CG). These routines are 
    are actually called vai the common user interface routines
    KSPSetType(), KSPSetFromOptions(), KSPSolve(), and KSPDestroy() so the
    application code interface remains identical for all preconditioners.

    Other basic routines for the KSP objects include
        KSPSetUp_XXX()
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
#include "src/ksp/ksp/impls/cg/cgctx.h"       /*I "petscksp.h" I*/
EXTERN PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP,PetscReal *,PetscReal *);
EXTERN PetscErrorCode KSPComputeEigenvalues_CG(KSP,PetscInt,PetscReal *,PetscReal *,PetscInt *);

/*
     KSPSetUp_CG - Sets up the workspace needed by the CG method. 

      This is called once, usually automatically by KSPSolve() or KSPSetUp()
     but can be called directly by KSPSetUp()
*/
#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_CG"
PetscErrorCode KSPSetUp_CG(KSP ksp)
{
  KSP_CG         *cgP = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;
  PetscInt        maxit = ksp->max_it;

  PetscFunctionBegin;
  /* 
       This implementation of CG only handles left preconditioning
     so generate an error otherwise.
  */
  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(2,"No right preconditioning for KSPCG");
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(2,"No symmetric preconditioning for KSPCG");
  }

  /* get work vectors needed by CG */
  ierr = KSPDefaultGetWork(ksp,3);CHKERRQ(ierr);

  /*
     If user requested computations of eigenvalues then allocate work
     work space needed
  */
  if (ksp->calc_sings) {
    /* get space to store tridiagonal matrix for Lanczos */
    ierr = PetscMalloc(2*(maxit+1)*sizeof(PetscScalar),&cgP->e);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(ksp,2*(maxit+1)*sizeof(PetscScalar));CHKERRQ(ierr);
    cgP->d                         = cgP->e + maxit + 1; 
    ierr = PetscMalloc(2*(maxit+1)*sizeof(PetscReal),&cgP->ee);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(ksp,2*(maxit+1)*sizeof(PetscScalar));CHKERRQ(ierr);
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
#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_CG"
PetscErrorCode  KSPSolve_CG(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,stored_max_it,eigs;
  PetscScalar    dpi,a = 1.0,beta,betaold = 1.0,b,*e = 0,*d = 0,mone = -1.0,ma;
  PetscReal      dp = 0.0;
  Vec            X,B,Z,R,P;
  KSP_CG         *cg;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  PetscTruth     diagonalscale;

  PetscFunctionBegin;
  ierr    = PCDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",ksp->type_name);

  cg            = (KSP_CG*)ksp->data;
  eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  P             = ksp->work[2];

#if !defined(PETSC_USE_COMPLEX)
#define VecXDot(x,y,a) VecDot(x,y,a)
#else
#define VecXDot(x,y,a) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x,y,a) : VecTDot(x,y,a))
#endif

  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; b = 0.0; }
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  ksp->its = 0;
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,R);CHKERRQ(ierr);             /*   r <- b - Ax       */
    ierr = VecAYPX(&mone,B,R);CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(B,R);CHKERRQ(ierr);                         /*     r <- b (x is 0) */
  }
  ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);                   /*     z <- Br         */
  ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);
  if (ksp->normtype == KSP_PRECONDITIONED_NORM) {
    ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);                /*    dp <- z'*z = e'*A'*B'*B*A'*e'   */
  } else if (ksp->normtype == KSP_UNPRECONDITIONED_NORM) {
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);                /*    dp <- r'*r = e'*A'*A*e         */
  } else if (ksp->normtype == KSP_NATURAL_NORM) {
    dp = sqrt(PetscAbsScalar(beta));                           /*    dp <- r'*z = r'*B*r = e'*A'*B*A*e */
  } else dp = 0.0;
  KSPLogResidualHistory(ksp,dp);
  KSPMonitor(ksp,0,dp);                              /* call any registered monitor routines */
  ksp->rnorm = dp;

  ierr = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);      /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
     ksp->its = i+1;
     ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);     /*     beta <- r'z     */
     if (beta == 0.0) {
       ksp->reason = KSP_CONVERGED_ATOL;
       ierr = PetscLogInfo((ksp,"KSPSolve_CG:converged due to beta = 0\n"));CHKERRQ(ierr);
       break;
#if !defined(PETSC_USE_COMPLEX)
     } else if (beta < 0.0) {
       ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
       ierr = PetscLogInfo((ksp,"KSPSolve_CG:diverging due to indefinite preconditioner\n"));CHKERRQ(ierr);
       break;
#endif
     }
     if (!i) {
       ierr = VecCopy(Z,P);CHKERRQ(ierr);         /*     p <- z          */
     } else {
         b = beta/betaold;
         if (eigs) {
	   if (ksp->max_it != stored_max_it) {
	     SETERRQ(PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
	   }
           e[i] = sqrt(PetscAbsScalar(b))/a;  
         }
         ierr = VecAYPX(&b,Z,P);CHKERRQ(ierr);    /*     p <- z + b* p   */
     }
     betaold = beta;
     ierr = KSP_MatMult(ksp,Amat,P,Z);CHKERRQ(ierr);      /*     z <- Kp         */
     ierr = VecXDot(P,Z,&dpi);CHKERRQ(ierr);      /*     dpi <- z'p      */
     a = beta/dpi;                                 /*     a = beta/p'z    */
     if (eigs) {
       d[i] = sqrt(PetscAbsScalar(b))*e[i] + 1.0/a;
     }
     ierr = VecAXPY(&a,P,X);CHKERRQ(ierr);          /*     x <- x + ap     */
     ma = -a; VecAXPY(&ma,Z,R);                      /*     r <- r - az     */
     if (ksp->normtype == KSP_PRECONDITIONED_NORM) {
       ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr);        /*     z <- Br         */
       ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z       */
     } else if (ksp->normtype == KSP_UNPRECONDITIONED_NORM) {
       ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- r'*r       */
     } else if (ksp->normtype == KSP_NATURAL_NORM) {
       dp = sqrt(PetscAbsScalar(beta));
     } else {
       dp = 0.0;
     }
     ksp->rnorm = dp;
     KSPLogResidualHistory(ksp,dp);
     KSPMonitor(ksp,i+1,dp);
     ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
     if (ksp->reason) break;
     if (ksp->normtype != KSP_PRECONDITIONED_NORM) {
       ierr = KSP_PCApply(ksp,R,Z);CHKERRQ(ierr); /* z <- Br  */
     }
     i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
}
/*
       KSPDestroy_CG - Frees all memory space used by the Krylov method

*/
#undef __FUNCT__  
#define __FUNCT__ "KSPDestroy_CG" 
PetscErrorCode KSPDestroy_CG(KSP ksp)
{
  KSP_CG         *cg = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* free space used for singular value calculations */
  if (ksp->calc_sings) {
    ierr = PetscFree(cg->e);CHKERRQ(ierr);
    ierr = PetscFree(cg->ee);CHKERRQ(ierr);
  }

  ierr = KSPDefaultFreeWork(ksp);CHKERRQ(ierr);
  
  /* free the context variable */
  ierr = PetscFree(cg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*
     KSPView_CG - Prints information about the current Krylov method being used

      Currently this only prints information to a file (or stdout) about the 
      symmetry of the problem. If your Krylov method has special options or 
      flags that information should be printed here.

*/
#undef __FUNCT__  
#define __FUNCT__ "KSPView_CG" 
PetscErrorCode KSPView_CG(KSP ksp,PetscViewer viewer)
{
#if defined(PETSC_USE_COMPLEX)
  KSP_CG         *cg = (KSP_CG *)ksp->data; 
  PetscErrorCode ierr;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (cg->type == KSP_CG_HERMITIAN) {
      ierr = PetscViewerASCIIPrintf(viewer,"  CG: variant for complex, Hermitian system\n");CHKERRQ(ierr);
    } else if (cg->type == KSP_CG_SYMMETRIC) {
      ierr = PetscViewerASCIIPrintf(viewer,"  CG: variant for complex, symmetric system\n");CHKERRQ(ierr);
    } else {
      ierr = PetscViewerASCIIPrintf(viewer,"  CG: unknown variant\n");CHKERRQ(ierr);
    }
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for KSP cg",((PetscObject)viewer)->type_name);
  }
#endif
  PetscFunctionReturn(0);
}

/*
    KSPSetFromOptions_CG - Checks the options database for options related to the 
                           conjugate gradient method.
*/ 
#undef __FUNCT__  
#define __FUNCT__ "KSPSetFromOptions_CG"
PetscErrorCode KSPSetFromOptions_CG(KSP ksp)
{
#if defined(PETSC_USE_COMPLEX)
  PetscErrorCode ierr;
  PetscTruth     flg;
#endif

  PetscFunctionBegin;
#if defined(PETSC_USE_COMPLEX)
  ierr = PetscOptionsHead("KSP CG options");CHKERRQ(ierr);
    ierr = PetscOptionsLogicalGroupBegin("-ksp_cg_Hermitian","Matrix is Hermitian","KSPCGSetType",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPCGSetType(ksp,KSP_CG_HERMITIAN);CHKERRQ(ierr); }
    ierr = PetscOptionsLogicalGroupEnd("-ksp_cg_symmetric","Matrix is complex symmetric, not Hermitian","KSPCGSetType",&flg);CHKERRQ(ierr);
    if (flg) { ierr = KSPCGSetType(ksp,KSP_CG_SYMMETRIC);CHKERRQ(ierr); }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
#endif
  PetscFunctionReturn(0);
}

/*
    KSPCGSetType_CG - This is an option that is SPECIFIC to this particular Krylov method.
                      This routine is registered below in KSPCreate_CG() and called from the 
                      routine KSPCGSetType() (see the file cgtype.c).

        This must be wrapped in an EXTERN_C_BEGIN to be dynamically linkable in C++
*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCGSetType_CG" 
PetscErrorCode KSPCGSetType_CG(KSP ksp,KSPCGType type)
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
/*MC
     KSPCG - The preconditioned conjugate gradient (PCG) iterative method

   Options Database Keys:
+   -ksp_cg_Hermitian - (for complex matrices only) indicates the matrix is Hermitian
-   -ksp_cg_symmetric - (for complex matrices only) indicates the matrix is symmetric

   Level: beginner

   Notes: The PCG method requires both the matrix and preconditioner to 
          be symmetric positive (semi) definite

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPCGSetType()

M*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_CG"
PetscErrorCode KSPCreate_CG(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CG         *cg;

  PetscFunctionBegin;
  ierr = PetscNew(KSP_CG,&cg);CHKERRQ(ierr);
  ierr = PetscLogObjectMemory(ksp,sizeof(KSP_CG));CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  cg->type                       = KSP_CG_SYMMETRIC;
#else
  cg->type                       = KSP_CG_HERMITIAN;
#endif
  ksp->data                      = (void*)cg;
  ksp->pc_side                   = PC_LEFT;

  /*
       Sets the functions that are associated with this data structure 
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup                = KSPSetUp_CG;
  ksp->ops->solve                = KSPSolve_CG;
  ksp->ops->destroy              = KSPDestroy_CG;
  ksp->ops->view                 = KSPView_CG;
  ksp->ops->setfromoptions       = KSPSetFromOptions_CG;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;

  /*
      Attach the function KSPCGSetType_CG() to this object. The routine 
      KSPCGSetType() checks for this attached function and calls it if it finds
      it. (Sort of like a dynamic member function that can be added at run time
  */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPCGSetType_C","KSPCGSetType_CG",
                                     KSPCGSetType_CG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END




