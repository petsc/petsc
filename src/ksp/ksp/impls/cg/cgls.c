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
  KSP_CGLS       *cgls = (KSP_CGLS*)ksp->data;

  PetscFunctionBegin;
  cgls->nwork_m = 2;
  if (cgls->vwork_m) {
    CHKERRQ(VecDestroyVecs(cgls->nwork_m,&cgls->vwork_m));
  }

  cgls->nwork_n = 2;
  if (cgls->vwork_n) {
    CHKERRQ(VecDestroyVecs(cgls->nwork_n,&cgls->vwork_n));
  }
  CHKERRQ(KSPCreateVecs(ksp,cgls->nwork_n,&cgls->vwork_n,cgls->nwork_m,&cgls->vwork_m));
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPSolve_CGLS(KSP ksp)
{
  KSP_CGLS       *cgls = (KSP_CGLS*)ksp->data;
  Mat            A;
  Vec            x,b,r,p,q,ss;
  PetscScalar    beta;
  PetscReal      alpha,gamma,oldgamma;

  PetscFunctionBegin;
  CHKERRQ(KSPGetOperators(ksp,&A,NULL)); /* Matrix of the system */

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
  CHKERRQ(MatMult(A,x,r));
  CHKERRQ(VecAYPX(r,-1,b));         /* r_0 = b - A * x_0  */
  CHKERRQ(KSP_MatMultHermitianTranspose(ksp,A,r,p)); /* p_0 = A^T * r_0    */
  CHKERRQ(VecCopy(p,ss));           /* s_0 = p_0          */
  CHKERRQ(VecNorm(ss,NORM_2,&gamma));
  KSPCheckNorm(ksp,gamma);
  if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = gamma;
  CHKERRQ((*ksp->converged)(ksp,ksp->its,ksp->rnorm,&ksp->reason,ksp->cnvP));
  if (ksp->reason) PetscFunctionReturn(0);
  gamma = gamma*gamma;                          /* gamma = norm2(s)^2 */

  do {
    CHKERRQ(MatMult(A,p,q));           /* q = A * p               */
    CHKERRQ(VecNorm(q,NORM_2,&alpha));
    KSPCheckNorm(ksp,alpha);
    alpha = alpha*alpha;                           /* alpha = norm2(q)^2      */
    alpha = gamma / alpha;                         /* alpha = gamma / alpha   */
    CHKERRQ(VecAXPY(x,alpha,p));       /* x += alpha * p          */
    CHKERRQ(VecAXPY(r,-alpha,q));      /* r -= alpha * q          */
    CHKERRQ(KSP_MatMultHermitianTranspose(ksp,A,r,ss)); /* ss = A^T * r            */
    oldgamma = gamma;                              /* oldgamma = gamma        */
    CHKERRQ(VecNorm(ss,NORM_2,&gamma));
    KSPCheckNorm(ksp,gamma);
    ksp->its++;
    if (ksp->normtype != KSP_NORM_NONE) ksp->rnorm = gamma;
    CHKERRQ(KSPMonitor(ksp,ksp->its,ksp->rnorm));
    CHKERRQ((*ksp->converged)(ksp,ksp->its,ksp->rnorm,&ksp->reason,ksp->cnvP));
    if (ksp->reason) PetscFunctionReturn(0);
    gamma = gamma*gamma;                           /* gamma = norm2(s)^2      */
    beta = gamma/oldgamma;                         /* beta = gamma / oldgamma */
    CHKERRQ(VecAYPX(p,beta,ss));       /* p = s + beta * p        */
  } while (ksp->its<ksp->max_it);

  if (ksp->its >= ksp->max_it) ksp->reason = KSP_DIVERGED_ITS;
  PetscFunctionReturn(0);
}

static PetscErrorCode KSPDestroy_CGLS(KSP ksp)
{
  KSP_CGLS       *cgls = (KSP_CGLS*)ksp->data;

  PetscFunctionBegin;
  /* Free work vectors */
  if (cgls->vwork_n) {
    CHKERRQ(VecDestroyVecs(cgls->nwork_n,&cgls->vwork_n));
  }
  if (cgls->vwork_m) {
    CHKERRQ(VecDestroyVecs(cgls->nwork_m,&cgls->vwork_m));
  }
  CHKERRQ(PetscFree(ksp->data));
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
  KSP_CGLS       *cgls;

  PetscFunctionBegin;
  CHKERRQ(PetscNewLog(ksp,&cgls));
  ksp->data                = (void*)cgls;
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,3));
  CHKERRQ(KSPSetSupportedNorm(ksp,KSP_NORM_NONE,PC_LEFT,1));
  ksp->ops->setup          = KSPSetUp_CGLS;
  ksp->ops->solve          = KSPSolve_CGLS;
  ksp->ops->destroy        = KSPDestroy_CGLS;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = NULL;
  ksp->ops->view           = NULL;
  PetscFunctionReturn(0);
}
