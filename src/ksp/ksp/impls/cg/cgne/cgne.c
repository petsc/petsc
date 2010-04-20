#define PETSCKSP_DLL

/*
       cgimpl.h defines the simple data structured used to store information
    related to the type of matrix (e.g. complex symmetric) being solved and
    data used during the optional Lanczo process used to compute eigenvalues
*/
#include "../src/ksp/ksp/impls/cg/cgimpl.h"       /*I "petscksp.h" I*/
EXTERN PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP,PetscReal *,PetscReal *);
EXTERN PetscErrorCode KSPComputeEigenvalues_CG(KSP,PetscInt,PetscReal *,PetscReal *,PetscInt *);


/*
     KSPSetUp_CGNE - Sets up the workspace needed by the CGNE method. 

     IDENTICAL TO THE CG ONE EXCEPT for one extra work vector!
*/
#undef __FUNCT__  
#define __FUNCT__ "KSPSetUp_CGNE"
PetscErrorCode KSPSetUp_CGNE(KSP ksp)
{
  KSP_CG         *cgP = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       maxit = ksp->max_it;

  PetscFunctionBegin;
  /* 
       This implementation of CGNE only handles left preconditioning
     so generate an error otherwise.
  */
  if (ksp->pc_side == PC_RIGHT) {
    SETERRQ(PETSC_ERR_SUP,"No right preconditioning for KSPCGNE");
  } else if (ksp->pc_side == PC_SYMMETRIC) {
    SETERRQ(PETSC_ERR_SUP,"No symmetric preconditioning for KSPCGNE");
  }

  /* get work vectors needed by CGNE */
  ierr = KSPDefaultGetWork(ksp,4);CHKERRQ(ierr);

  /*
     If user requested computations of eigenvalues then allocate work
     work space needed
  */
  if (ksp->calc_sings) {
    /* get space to store tridiagonal matrix for Lanczos */
    ierr = PetscMalloc4(maxit+1,PetscScalar,&cgP->e,maxit+1,PetscScalar,&cgP->d,maxit+1,PetscReal,&cgP->ee,maxit+1,PetscReal,&cgP->dd);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory(ksp,2*(maxit+1)*(sizeof(PetscScalar)+sizeof(PetscReal)));CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*
       KSPSolve_CGNE - This routine actually applies the conjugate gradient 
    method

   Input Parameter:
.     ksp - the Krylov space object that was set to use conjugate gradient, by, for 
            example, KSPCreate(MPI_Comm,KSP *ksp); KSPSetType(ksp,KSPCG);


    Virtually identical to the KSPSolve_CG, it should definitely reuse the same code.

*/
#undef __FUNCT__  
#define __FUNCT__ "KSPSolve_CGNE"
PetscErrorCode  KSPSolve_CGNE(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,stored_max_it,eigs;
  PetscScalar    dpi,a = 1.0,beta,betaold = 1.0,b = 0,*e = 0,*d = 0;
  PetscReal      dp = 0.0;
  Vec            X,B,Z,R,P,T;
  KSP_CG         *cg;
  Mat            Amat,Pmat;
  MatStructure   pflag;
  PetscTruth     diagonalscale,transpose_pc;

  PetscFunctionBegin;
  ierr    = PCDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);
  ierr = PCApplyTransposeExists(ksp->pc,&transpose_pc);CHKERRQ(ierr);

  cg            = (KSP_CG*)ksp->data;
  eigs          = ksp->calc_sings;
  stored_max_it = ksp->max_it;
  X             = ksp->vec_sol;
  B             = ksp->vec_rhs;
  R             = ksp->work[0];
  Z             = ksp->work[1];
  P             = ksp->work[2];
  T             = ksp->work[3];

#if !defined(PETSC_USE_COMPLEX)
#define VecXDot(x,y,a) VecDot(x,y,a)
#else
#define VecXDot(x,y,a) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x,y,a) : VecTDot(x,y,a))
#endif

  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; }
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat,&pflag);CHKERRQ(ierr);

  ksp->its = 0;
  ierr = MatMultTranspose(Amat,B,T);CHKERRQ(ierr);
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,P);CHKERRQ(ierr);
    ierr = KSP_MatMultTranspose(ksp,Amat,P,R);CHKERRQ(ierr);
    ierr = VecAYPX(R,-1.0,T);CHKERRQ(ierr);
  } else { 
    ierr = VecCopy(T,R);CHKERRQ(ierr);              /*     r <- b (x is 0) */
  }
  ierr = KSP_PCApply(ksp,R,T);CHKERRQ(ierr);
  if (transpose_pc) {
    ierr = KSP_PCApplyTranspose(ksp,T,Z);CHKERRQ(ierr);
  } else {
    ierr = KSP_PCApply(ksp,T,Z);CHKERRQ(ierr);
  }

  if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
    ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr); /*    dp <- z'*z       */
  } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr); /*    dp <- r'*r       */
  } else if (ksp->normtype == KSP_NORM_NATURAL) {
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);
    dp = sqrt(PetscAbsScalar(beta));
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
       ierr = PetscInfo(ksp,"converged due to beta = 0\n");CHKERRQ(ierr);
       break;
#if !defined(PETSC_USE_COMPLEX)
     } else if (beta < 0.0) {
       ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
       ierr = PetscInfo(ksp,"diverging due to indefinite preconditioner\n");CHKERRQ(ierr);
       break;
#endif
     }
     if (!i) {
       ierr = VecCopy(Z,P);CHKERRQ(ierr);         /*     p <- z          */
       b = 0.0;
     } else {
       b = beta/betaold;
       if (eigs) {
	 if (ksp->max_it != stored_max_it) {
	   SETERRQ(PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
	 }
	 e[i] = sqrt(PetscAbsScalar(b))/a;  
       }
       ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);    /*     p <- z + b* p   */
     }
     betaold = beta;
     ierr = MatMult(Amat,P,T);CHKERRQ(ierr);
     ierr = MatMultTranspose(Amat,T,Z);CHKERRQ(ierr);
     ierr = VecXDot(P,Z,&dpi);CHKERRQ(ierr);      /*     dpi <- z'p      */
     a = beta/dpi;                                 /*     a = beta/p'z    */
     if (eigs) {
       d[i] = sqrt(PetscAbsScalar(b))*e[i] + 1.0/a;
     }
     ierr = VecAXPY(X,a,P);CHKERRQ(ierr);          /*     x <- x + ap     */
     ierr = VecAXPY(R,-a,Z);CHKERRQ(ierr);                      /*     r <- r - az     */
     if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
       ierr = KSP_PCApply(ksp,R,T);CHKERRQ(ierr);
       if (transpose_pc) {
	 ierr = KSP_PCApplyTranspose(ksp,T,Z);CHKERRQ(ierr);
       } else {
	 ierr = KSP_PCApply(ksp,T,Z);CHKERRQ(ierr);
       }
       ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z       */
     } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
       ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
     } else if (ksp->normtype == KSP_NORM_NATURAL) {
       dp = sqrt(PetscAbsScalar(beta));
     } else {
       dp = 0.0;
     }
     ksp->rnorm = dp;
     KSPLogResidualHistory(ksp,dp);
     KSPMonitor(ksp,i+1,dp);
     ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
     if (ksp->reason) break;
     if (ksp->normtype != KSP_NORM_PRECONDITIONED) {
       if (transpose_pc) {
	 ierr = KSP_PCApplyTranspose(ksp,T,Z);CHKERRQ(ierr);
       } else {
	 ierr = KSP_PCApply(ksp,T,Z);CHKERRQ(ierr);
       }
     }
     i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) {
    ksp->reason = KSP_DIVERGED_ITS;
  }
  PetscFunctionReturn(0);
}

/*
    KSPCreate_CGNE - Creates the data structure for the Krylov method CGNE and sets the 
       function pointers for all the routines it needs to call (KSPSolve_CGNE() etc)

    It must be wrapped in EXTERN_C_BEGIN to be dynamically linkable in C++
*/

/*MC
     KSPCGNE - Applies the preconditioned conjugate gradient method to the normal equations
          without explicitly forming A^t*A

   Options Database Keys:
.   -ksp_cg_type <Hermitian or symmetric - (for complex matrices only) indicates the matrix is Hermitian or symmetric


   Level: beginner

   Notes: eigenvalue computation routines will return information about the
          spectrum of A^t*A, rather than A.

   This is NOT a different algorithm then used with KSPCG, it merely uses that algorithm with the 
   matrix defined by A^t*A and preconditioner defined by B^t*B where B is the preconditioner for A.

   This method requires that one be apply to apply the transpose of the preconditioner and operator
   as well as the operator and preconditioner. If the transpose of the preconditioner is not available then
   the preconditioner is used in its place so one ends up preconditioning A'A with B B. Seems odd?

   This only supports left preconditioning.

   Developer Notes: How is this related to the preconditioned LSQR implementation?

   This object is subclassed off of KSPCG

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPCGSetType(), KSPBICG

M*/

extern PetscErrorCode KSPDestroy_CG(KSP);
extern PetscErrorCode KSPView_CG(KSP,PetscViewer);
extern PetscErrorCode KSPSetFromOptions_CG(KSP);
EXTERN_C_BEGIN
extern PetscErrorCode PETSCKSP_DLLEXPORT KSPCGSetType_CG(KSP,KSPCGType);
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "KSPCreate_CGNE"
PetscErrorCode PETSCKSP_DLLEXPORT KSPCreate_CGNE(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CG         *cg;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,KSP_CG,&cg);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  cg->type                       = KSP_CG_SYMMETRIC;
#else
  cg->type                       = KSP_CG_HERMITIAN;
#endif
  ksp->data                      = (void*)cg;
  if (ksp->pc_side != PC_LEFT) {
     ierr = PetscInfo(ksp,"WARNING! Setting PC_SIDE for CGNE to left!\n");CHKERRQ(ierr);
  }
  ksp->pc_side                   = PC_LEFT;

  /*
       Sets the functions that are associated with this data structure 
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup                = KSPSetUp_CGNE;
  ksp->ops->solve                = KSPSolve_CGNE;
  ksp->ops->destroy              = KSPDestroy_CG;
  ksp->ops->view                 = KSPView_CG;
  ksp->ops->setfromoptions       = KSPSetFromOptions_CG;
  ksp->ops->buildsolution        = KSPDefaultBuildSolution;
  ksp->ops->buildresidual        = KSPDefaultBuildResidual;

  /*
      Attach the function KSPCGSetType_CGNE() to this object. The routine 
      KSPCGSetType() checks for this attached function and calls it if it finds
      it. (Sort of like a dynamic member function that can be added at run time
  */
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)ksp,"KSPCGSetType_C","KSPCGSetType_CG",KSPCGSetType_CG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END




