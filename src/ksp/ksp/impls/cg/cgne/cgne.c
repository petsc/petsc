
/*
       cgimpl.h defines the simple data structured used to store information
    related to the type of matrix (e.g. complex symmetric) being solved and
    data used during the optional Lanczo process used to compute eigenvalues
*/
#include <../src/ksp/ksp/impls/cg/cgimpl.h>       /*I "petscksp.h" I*/
extern PetscErrorCode KSPComputeExtremeSingularValues_CG(KSP,PetscReal*,PetscReal*);
extern PetscErrorCode KSPComputeEigenvalues_CG(KSP,PetscInt,PetscReal*,PetscReal*,PetscInt*);

static PetscErrorCode  KSPCGSetType_CGNE(KSP ksp,KSPCGType type)
{
  KSP_CG *cg = (KSP_CG*)ksp->data;

  PetscFunctionBegin;
  cg->type = type;
  PetscFunctionReturn(0);
}

/*
     KSPSetUp_CGNE - Sets up the workspace needed by the CGNE method.

     IDENTICAL TO THE CG ONE EXCEPT for one extra work vector!
*/
static PetscErrorCode KSPSetUp_CGNE(KSP ksp)
{
  KSP_CG         *cgP = (KSP_CG*)ksp->data;
  PetscErrorCode ierr;
  PetscInt       maxit = ksp->max_it;

  PetscFunctionBegin;
  /* get work vectors needed by CGNE */
  ierr = KSPSetWorkVecs(ksp,4);CHKERRQ(ierr);

  /*
     If user requested computations of eigenvalues then allocate work
     work space needed
  */
  if (ksp->calc_sings) {
    /* get space to store tridiagonal matrix for Lanczos */
    ierr = PetscMalloc4(maxit+1,&cgP->e,maxit+1,&cgP->d,maxit+1,&cgP->ee,maxit+1,&cgP->dd);CHKERRQ(ierr);
    ierr = PetscLogObjectMemory((PetscObject)ksp,2*(maxit+1)*(sizeof(PetscScalar)+sizeof(PetscReal)));CHKERRQ(ierr);

    ksp->ops->computeextremesingularvalues = KSPComputeExtremeSingularValues_CG;
    ksp->ops->computeeigenvalues           = KSPComputeEigenvalues_CG;
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
static PetscErrorCode  KSPSolve_CGNE(KSP ksp)
{
  PetscErrorCode ierr;
  PetscInt       i,stored_max_it,eigs;
  PetscScalar    dpi,a = 1.0,beta,betaold = 1.0,b = 0,*e = NULL,*d = NULL;
  PetscReal      dp = 0.0;
  Vec            X,B,Z,R,P,T;
  KSP_CG         *cg;
  Mat            Amat,Pmat;
  PetscBool      diagonalscale,transpose_pc;

  PetscFunctionBegin;
  ierr = PCGetDiagonalScale(ksp->pc,&diagonalscale);CHKERRQ(ierr);
  if (diagonalscale) SETERRQ1(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Krylov method %s does not support diagonal scaling",((PetscObject)ksp)->type_name);
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

#define VecXDot(x,y,a) (((cg->type) == (KSP_CG_HERMITIAN)) ? VecDot(x,y,a) : VecTDot(x,y,a))

  if (eigs) {e = cg->e; d = cg->d; e[0] = 0.0; }
  ierr = PCGetOperators(ksp->pc,&Amat,&Pmat);CHKERRQ(ierr);

  ksp->its = 0;
  ierr     = KSP_MatMultTranspose(ksp,Amat,B,T);CHKERRQ(ierr);
  if (!ksp->guess_zero) {
    ierr = KSP_MatMult(ksp,Amat,X,P);CHKERRQ(ierr);
    ierr = KSP_MatMultTranspose(ksp,Amat,P,R);CHKERRQ(ierr);
    ierr = VecAYPX(R,-1.0,T);CHKERRQ(ierr);
  } else {
    ierr = VecCopy(T,R);CHKERRQ(ierr);              /*     r <- b (x is 0) */
  }
  if (transpose_pc) {
    ierr = KSP_PCApplyTranspose(ksp,R,T);CHKERRQ(ierr);
  } else {
    ierr = KSP_PCApply(ksp,R,T);CHKERRQ(ierr);
  }
  ierr = KSP_PCApply(ksp,T,Z);CHKERRQ(ierr);

  if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
    ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr); /*    dp <- z'*z       */
  } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
    ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr); /*    dp <- r'*r       */
  } else if (ksp->normtype == KSP_NORM_NATURAL) {
    ierr = VecXDot(Z,R,&beta);CHKERRQ(ierr);
    KSPCheckDot(ksp,beta);
    dp   = PetscSqrtReal(PetscAbsScalar(beta));
  } else dp = 0.0;
  ierr       = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
  ierr       = KSPMonitor(ksp,0,dp);CHKERRQ(ierr);
  ksp->rnorm = dp;
  ierr       = (*ksp->converged)(ksp,0,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr); /* test for convergence */
  if (ksp->reason) PetscFunctionReturn(0);

  i = 0;
  do {
    ksp->its = i+1;
    ierr     = VecXDot(Z,R,&beta);CHKERRQ(ierr); /*     beta <- r'z     */
    KSPCheckDot(ksp,beta);
    if (beta == 0.0) {
      ksp->reason = KSP_CONVERGED_ATOL;
      ierr        = PetscInfo(ksp,"converged due to beta = 0\n");CHKERRQ(ierr);
      break;
#if !defined(PETSC_USE_COMPLEX)
    } else if (beta < 0.0) {
      ksp->reason = KSP_DIVERGED_INDEFINITE_PC;
      ierr        = PetscInfo(ksp,"diverging due to indefinite preconditioner\n");CHKERRQ(ierr);
      break;
#endif
    }
    if (!i) {
      ierr = VecCopy(Z,P);CHKERRQ(ierr);          /*     p <- z          */
      b    = 0.0;
    } else {
      b = beta/betaold;
      if (eigs) {
        if (ksp->max_it != stored_max_it) SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"Can not change maxit AND calculate eigenvalues");
        e[i] = PetscSqrtReal(PetscAbsScalar(b))/a;
      }
      ierr = VecAYPX(P,b,Z);CHKERRQ(ierr);     /*     p <- z + b* p   */
    }
    betaold = beta;
    ierr    = KSP_MatMult(ksp,Amat,P,T);CHKERRQ(ierr);
    ierr    = KSP_MatMultTranspose(ksp,Amat,T,Z);CHKERRQ(ierr);
    ierr    = VecXDot(P,Z,&dpi);CHKERRQ(ierr);    /*     dpi <- z'p      */
    KSPCheckDot(ksp,dpi);
    a       = beta/dpi;                            /*     a = beta/p'z    */
    if (eigs) d[i] = PetscSqrtReal(PetscAbsScalar(b))*e[i] + 1.0/a;
    ierr = VecAXPY(X,a,P);CHKERRQ(ierr);           /*     x <- x + ap     */
    ierr = VecAXPY(R,-a,Z);CHKERRQ(ierr);                       /*     r <- r - az     */
    if (ksp->normtype == KSP_NORM_PRECONDITIONED) {
      if (transpose_pc) {
        ierr = KSP_PCApplyTranspose(ksp,R,T);CHKERRQ(ierr);
      } else {
        ierr = KSP_PCApply(ksp,R,T);CHKERRQ(ierr);
      }
      ierr = KSP_PCApply(ksp,T,Z);CHKERRQ(ierr);
      ierr = VecNorm(Z,NORM_2,&dp);CHKERRQ(ierr);              /*    dp <- z'*z       */
    } else if (ksp->normtype == KSP_NORM_UNPRECONDITIONED) {
      ierr = VecNorm(R,NORM_2,&dp);CHKERRQ(ierr);
    } else if (ksp->normtype == KSP_NORM_NATURAL) {
      dp = PetscSqrtReal(PetscAbsScalar(beta));
    } else {
      dp = 0.0;
    }
    ksp->rnorm = dp;
    ierr = KSPLogResidualHistory(ksp,dp);CHKERRQ(ierr);
    if (eigs) cg->ned = ksp->its;
    ierr = KSPMonitor(ksp,i+1,dp);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,i+1,dp,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) break;
    if (ksp->normtype != KSP_NORM_PRECONDITIONED) {
      if (transpose_pc) {
        ierr = KSP_PCApplyTranspose(ksp,R,T);CHKERRQ(ierr);
      } else {
        ierr = KSP_PCApply(ksp,R,T);CHKERRQ(ierr);
      }
      ierr = KSP_PCApply(ksp,T,Z);CHKERRQ(ierr);
    }
    i++;
  } while (i<ksp->max_it);
  if (i >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

/*
    KSPCreate_CGNE - Creates the data structure for the Krylov method CGNE and sets the
       function pointers for all the routines it needs to call (KSPSolve_CGNE() etc)

    It must be labeled as PETSC_EXTERN to be dynamically linkable in C++
*/

/*MC
     KSPCGNE - Applies the preconditioned conjugate gradient method to the normal equations
          without explicitly forming A^t*A

   Options Database Keys:
.   -ksp_cg_type <Hermitian or symmetric - (for complex matrices only) indicates the matrix is Hermitian or symmetric


   Level: beginner

   Notes:
    eigenvalue computation routines will return information about the
          spectrum of A^t*A, rather than A.


   CGNE is a general-purpose non-symmetric method. It works well when the singular values are much better behaved than
   eigenvalues. A unitary matrix is a classic example where CGNE converges in one iteration, but GMRES and CGS need N
   iterations (see Nachtigal, Reddy, and Trefethen, "How fast are nonsymmetric matrix iterations", 1992). If you intend
   to solve least squares problems, use KSPLSQR.

   This is NOT a different algorithm than used with KSPCG, it merely uses that algorithm with the
   matrix defined by A^t*A and preconditioner defined by B^t*B where B is the preconditioner for A.

   This method requires that one be able to apply the transpose of the preconditioner and operator
   as well as the operator and preconditioner. If the transpose of the preconditioner is not available then
   the preconditioner is used in its place so one ends up preconditioning A'A with B B. Seems odd?

   This only supports left preconditioning.

   This object is subclassed off of KSPCG

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPCGSetType(), KSPBICG

M*/

PETSC_EXTERN PetscErrorCode KSPCreate_CGNE(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CG         *cg;

  PetscFunctionBegin;
  ierr = PetscNewLog(ksp,&cg);CHKERRQ(ierr);
#if !defined(PETSC_USE_COMPLEX)
  cg->type = KSP_CG_SYMMETRIC;
#else
  cg->type = KSP_CG_HERMITIAN;
#endif
  ksp->data = (void*)cg;
  ierr      = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr      = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr      = KSPSetSupportedNorm(ksp,KSP_NORM_NATURAL,PC_LEFT,2);CHKERRQ(ierr);

  /*
       Sets the functions that are associated with this data structure
       (in C++ this is the same as defining virtual functions)
  */
  ksp->ops->setup          = KSPSetUp_CGNE;
  ksp->ops->solve          = KSPSolve_CGNE;
  ksp->ops->destroy        = KSPDestroy_CG;
  ksp->ops->view           = KSPView_CG;
  ksp->ops->setfromoptions = KSPSetFromOptions_CG;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;

  /*
      Attach the function KSPCGSetType_CGNE() to this object. The routine
      KSPCGSetType() checks for this attached function and calls it if it finds
      it. (Sort of like a dynamic member function that can be added at run time
  */
  ierr = PetscObjectComposeFunction((PetscObject)ksp,"KSPCGSetType_C",KSPCGSetType_CGNE);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}




