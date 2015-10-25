/*
    This file implements CGLS, the Conjugate Gradient method for Least-Squares problems.
*/
#include <petsc/private/kspimpl.h>	/*I "petscksp.h" I*/
#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_CGLS"
static PetscErrorCode KSPSetUp_CGLS(KSP ksp)
{
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_CGLS"
PetscErrorCode KSPSolve_CGLS(KSP ksp)
{
  PetscErrorCode ierr;
  Mat            A;
  Vec            x,b,r,p,q,ss;
  PetscScalar    alpha,beta,gamma,oldgamma,rnorm;
  PetscInt       maxiter_ls = 15;
  
  PetscFunctionBegin;
  ierr = KSPGetRhs(ksp,&b);CHKERRQ(ierr);            /* Right-hand side vector */
  ierr = KSPGetOperators(ksp,&A,NULL);CHKERRQ(ierr); /* Matrix of the system */
  ierr = KSPGetSolution(ksp,&x);CHKERRQ(ierr);       /* Solution vector */
  
  ierr = VecDuplicate(x,&p);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&ss);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&q);CHKERRQ(ierr);
  
  /* Minimization with the CGLS method */
  ksp->its = 0;
  MatMult(A,x,r); VecAYPX(r,-1,b);                /* r_0 = b - A * x_0  */
  MatMultTranspose(A,r,p);                        /* p_0 = A^T * r_0    */
  ierr = VecCopy(p,ss);CHKERRQ(ierr);             /* s_0 = p_0          */
  VecNorm(ss,NORM_2,&gamma); gamma = gamma*gamma; /* gamma = norm2(s)^2 */
  
  do {
    MatMult(A,p,q);                                 /* q = A * p               */
    VecNorm(q,NORM_2,&alpha); alpha = alpha*alpha;  /* alpha = norm2(q)^2      */
    alpha = gamma / alpha;                          /* alpha = gamma / alpha   */
    VecAXPY(x,alpha,p);                             /* x += alpha * p          */
    VecAXPY(r,-alpha,q);                            /* r -= alpha * q          */
    MatMultTranspose(A,r,ss);                       /* ss = A^T * r            */
    oldgamma = gamma;                               /* oldgamma = gamma        */
    VecNorm(ss,NORM_2,&gamma); gamma = gamma*gamma; /* gamma = norm2(s)^2      */
    beta = gamma/oldgamma;                          /* beta = gamma / oldgamma */
    VecAYPX(p,beta,ss);                             /* p = s + beta * p        */
    ksp->its ++;
    ksp->rnorm = gamma;
  } while (ksp->its<ksp->max_it);
  
  MatMult(A,x,r);
  VecAXPY(r,-1,b);
  VecNorm(r,NORM_2,&rnorm);
  ksp->rnorm = rnorm;
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nCGLS : iter %d, norm %.2e\n\n",ksp->its,ksp->rnorm);CHKERRQ(ierr);
  ierr = (*ksp->converged)(ksp,ksp->its,rnorm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  if (ksp->its>=maxiter_ls && !ksp->reason) ksp->reason = KSP_DIVERGED_ITS;

  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&p);CHKERRQ(ierr);
  ierr = VecDestroy(&q);CHKERRQ(ierr);
  ierr = VecDestroy(&ss);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPCreate_CGLS"
PETSC_EXTERN PetscErrorCode KSPCreate_CGLS(KSP ksp)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ksp->data                = (void*)0;
  ierr                     = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_LEFT,3);CHKERRQ(ierr);
  ksp->ops->setup          = KSPSetUp_CGLS;
  ksp->ops->solve          = KSPSolve_CGLS;
  ksp->ops->destroy        = KSPDestroyDefault;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = 0;
  ksp->ops->view           = 0;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"This is not supported for complex numbers");
#endif
  PetscFunctionReturn(0);
}
