/*
    This file implements CGLS, the Conjugate Gradient method for Least-Squares problems.
*/
#include <petsc/private/kspimpl.h>      /*I "petscksp.h" I*/

typedef struct {
  PetscInt  nwork_n,nwork_m;
  Vec       *vwork_m;   /* work vectors of length m, where the system is size m x n */
  Vec       *vwork_n;   /* work vectors of length n */
} KSP_CGLS;

static PetscErrorCode KSPSetUp_CGLS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CGLS       *cgls = (KSP_CGLS*)ksp->data;

  PetscFunctionBegin;
  cgls->nwork_m = 2;
  if (cgls->vwork_m) {
    ierr = VecDestroyVecs(cgls->nwork_m,&cgls->vwork_m);CHKERRQ(ierr);
  }

  cgls->nwork_n = 2;
  if (cgls->vwork_n) {
    ierr = VecDestroyVecs(cgls->nwork_n,&cgls->vwork_n);CHKERRQ(ierr);
  }
  ierr = KSPCreateVecs(ksp,cgls->nwork_n,&cgls->vwork_n,cgls->nwork_m,&cgls->vwork_m);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_CGLS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CGLS       *cgls = (KSP_CGLS*)ksp->data;
  Mat            A;
  Vec            x,b,r,p,q,ss;
  PetscScalar    beta;
  PetscReal      alpha,gamma,oldgamma;

  PetscFunctionBegin;
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr); /* Matrix of the system */

  /* vectors of length n, where system size is mxn */
  x  = ksp->vec_sol; /* Solution vector */
  p  = cgls->vwork_n[0];
  ss  = cgls->vwork_n[1];

  /* vectors of length m, where system size is mxn */
  b  = ksp->vec_rhs; /* Right-hand side vector */
  r  = cgls->vwork_m[0];
  q  = cgls->vwork_m[1];

  /* Minimization with the CGLS method */
  ksp->its = 0;
  ksp->rnorm = 0;
  ierr = MatMult(A,x,r);CHKERRQ(ierr);
  ierr = VecAYPX(r,-1,b);CHKERRQ(ierr);         /* r_0 = b - A * x_0  */
  ierr = MatMultTranspose(A,r,p);CHKERRQ(ierr); /* p_0 = A^T * r_0    */
  ierr = VecCopy(p,ss);CHKERRQ(ierr);           /* s_0 = p_0          */
  ierr = VecNorm(ss,NORM_2,&gamma);CHKERRQ(ierr);
  KSPCheckNorm(ksp,gamma);
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = gamma;
  ierr = (*ksp->converged)(ksp,ksp->its,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->reason) PetscFunctionReturn(0);
  gamma = gamma*gamma;                          /* gamma = norm2(s)^2 */

  do {
    ierr = MatMult(A,p,q);CHKERRQ(ierr);           /* q = A * p               */
    ierr = VecNorm(q,NORM_2,&alpha);CHKERRQ(ierr);
    KSPCheckNorm(ksp,alpha);
    alpha = alpha*alpha;                           /* alpha = norm2(q)^2      */
    alpha = gamma / alpha;                         /* alpha = gamma / alpha   */
    ierr = VecAXPY(x,alpha,p);CHKERRQ(ierr);       /* x += alpha * p          */
    ierr = VecAXPY(r,-alpha,q);CHKERRQ(ierr);      /* r -= alpha * q          */
    ierr = MatMultTranspose(A,r,ss);CHKERRQ(ierr); /* ss = A^T * r            */
    oldgamma = gamma;                              /* oldgamma = gamma        */
    ierr = VecNorm(ss,NORM_2,&gamma);CHKERRQ(ierr);
    KSPCheckNorm(ksp,gamma);
    ksp->its++;
    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = gamma;
    ierr = KSPMonitor(ksp,ksp->its,ksp->rnorm);CHKERRQ(ierr);
    ierr = (*ksp->converged)(ksp,ksp->its,ksp->rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
    if (ksp->reason) PetscFunctionReturn(0);
    gamma = gamma*gamma;                           /* gamma = norm2(s)^2      */
    beta = gamma/oldgamma;                         /* beta = gamma / oldgamma */
    ierr = VecAYPX(p,beta,ss);CHKERRQ(ierr);       /* p = s + beta * p        */
  } while (ksp->its<ksp->max_it);

  if (ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_CGLS(KSP ksp)
{
  KSP_CGLS       *cgls = (KSP_CGLS*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Free work vectors */
  if (cgls->vwork_n) {
    ierr = VecDestroyVecs(cgls->nwork_n,&cgls->vwork_n);CHKERRQ(ierr);
  }
  if (cgls->vwork_m) {
    ierr = VecDestroyVecs(cgls->nwork_m,&cgls->vwork_m);CHKERRQ(ierr);
  }
  ierr = PetscFree(ksp->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     KSPCGLS - Conjugate Gradient method for Least-Squares problems

   Level: beginner

   Supports non-square (rectangular) matrices.

   Notes:
    This does not use the preconditioner, so one should probably use KSPLSQR instead.

.seealso:  KSPCreate(), KSPSetType(), KSPType (for list of available types), KSP,
           KSPCGSetType(), KSPCGUseSingleReduction(), KSPPIPECG, KSPGROPPCG

M*/
PETSC_EXTERN PetscErrorCode KSPCreate_CGLS(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_CGLS       *cgls;

  PetscFunctionBegin;
  ierr                     = PetscNewLog(ksp,&cgls);CHKERRQ(ierr);
  ksp->data                = (void*)cgls;
  ierr                     = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ierr                     = KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1);CHKERRQ(ierr);
  ksp->ops->setup          = KSPSetUp_CGLS;
  ksp->ops->solve          = KSPSolve_CGLS;
  ksp->ops->destroy        = KSPDestroy_CGLS;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"This is not supported for complex numbers");
#endif
  PetscFunctionReturn(0);
}
