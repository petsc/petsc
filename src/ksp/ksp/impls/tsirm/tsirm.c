/*
    This file implements TSIRM, the Two-Stage Iteration with least-squares Residual Minimization method. 
    It is an iterative method to solve large sparse linear systems of the form Ax=b, and it improves the convergence of Krylov based iterative methods.
    The principle is to build an external iteration over a Krylov method (for example GMRES), and to frequently store its current residual in a matrix S. After a given number of outer iterations, a least-squares minimization step (with CGLS or LSQR) is applied on S, in order to compute a better solution and to make new iterations if required.
*/
#include <petsc/private/kspimpl.h>	/*I "petscksp.h" I*/
#include <petscksp.h>

typedef struct {
  PetscReal tol_ls;
  PetscInt  size_ls,maxiter_ls,cgls,size,Istart,Iend;
  Mat       A,S;
  Vec       Alpha,r;
} KSP_TSIRM;

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_TSIRM"
static PetscErrorCode KSPSetUp_TSIRM(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_TSIRM      *tsirm = (KSP_TSIRM*)ksp->data;
  
  PetscFunctionBegin;
  /* Initialization */
  tsirm->tol_ls     = 1e-40;
  tsirm->size_ls    = 12;
  tsirm->maxiter_ls = 15;
  tsirm->cgls       = 0;
  
  /* Matrix of the system */
  ierr = KSPGetOperators(ksp,&tsirm->A,NULL);CHKERRQ(ierr);    /* Matrix of the system   */
  ierr = MatGetSize(tsirm->A,&tsirm->size,NULL);CHKERRQ(ierr); /* Size of the system     */
  ierr = MatGetOwnershipRange(tsirm->A,&tsirm->Istart,&tsirm->Iend);CHKERRQ(ierr);
  
  /* Matrix S of residuals */
  ierr = MatCreate(PETSC_COMM_WORLD,&tsirm->S);CHKERRQ(ierr); 
  ierr = MatSetSizes(tsirm->S,tsirm->Iend-tsirm->Istart,PETSC_DECIDE,tsirm->size,tsirm->size_ls);CHKERRQ(ierr);
  ierr = MatSetType(tsirm->S,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(tsirm->S);CHKERRQ(ierr);
  
  /* Residual and vector Alpha computed in the minimization step */
  ierr = MatCreateVecs(tsirm->S,&tsirm->Alpha,&tsirm->r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_TSIRM"
PetscErrorCode KSPSolve_TSIRM(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_TSIRM      *tsirm = (KSP_TSIRM*)ksp->data;
  KSP            sub_ksp;
  PC             pc;
  Mat            AS;
  Vec            x,b;
  PetscScalar    *array;
  PetscReal      norm = 20;
  PetscInt       i,*ind_row,first_iteration = 1,its = 0,total = 0,col = 0;
  PetscInt       iter_minimization = 0,restart = 30;
  KSP            ksp_min;  /* KSP for minimization */
  PC             pc_min;    /* PC for minimization */
  
  PetscFunctionBegin;  
  x = ksp->vec_sol; /* Solution vector        */
  b = ksp->vec_rhs; /* Right-hand side vector */
  
  /* Row indexes (these indexes are global) */
  ierr = PetscMalloc(tsirm->Iend-tsirm->Istart,&ind_row);CHKERRQ(ierr);
  for (i=0;i<tsirm->Iend-tsirm->Istart;i++) ind_row[i] = i+tsirm->Istart;
  
  /* Inner solver */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = PCKSPGetKSP(pc,&sub_ksp);CHKERRQ(ierr);
  ierr = KSPSetTolerances(sub_ksp,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT,restart);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(sub_ksp);CHKERRQ(ierr);
  
  /* previously it seemed good but with SNES it seems not good... */
  ierr = KSP_MatMult(sub_ksp,tsirm->A,x,tsirm->r);CHKERRQ(ierr);
  ierr = VecAXPY(tsirm->r,-1,b);CHKERRQ(ierr);
  ierr = VecNorm(tsirm->r,NORM_2,&norm);CHKERRQ(ierr);
  ksp->its = 0;
  ierr = KSPConvergedDefault(ksp,ksp->its,norm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(sub_ksp,PETSC_TRUE);CHKERRQ(ierr);
  do {
    for (col=0;col<tsirm->size_ls && ksp->reason==0;col++) {
      /* Solve (inner ietration) */
      ierr = KSPSolve(sub_ksp,b,x);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(sub_ksp,&its);CHKERRQ(ierr);
      total += its;
      
      /* Build S^T */
      ierr = VecGetArray(x,&array);CHKERRQ(ierr);
      ierr = MatSetValues(tsirm->S,tsirm->Iend-tsirm->Istart,ind_row,1,&col,array,INSERT_VALUES);CHKERRQ(ierr);
      ierr = VecRestoreArray(x,&array);CHKERRQ(ierr);
      
      ierr = KSPGetResidualNorm(sub_ksp,&norm);CHKERRQ(ierr);
      ksp->rnorm = norm;
      ksp->its ++;
      ierr = KSPConvergedDefault(ksp,ksp->its,norm,&ksp->reason,ksp->cnvP);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp,ksp->its,norm);CHKERRQ(ierr);
    }
    
    /* Minimization step */
    if (!ksp->reason) {
      ierr = MatAssemblyBegin(tsirm->S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      ierr = MatAssemblyEnd(tsirm->S,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
      if (first_iteration) {
        ierr = MatMatMult(tsirm->A,tsirm->S,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AS);CHKERRQ(ierr);
        first_iteration = 0;
      } else {
        ierr = MatMatMult(tsirm->A,tsirm->S,MAT_REUSE_MATRIX,PETSC_DEFAULT,&AS);CHKERRQ(ierr);
      }
      
      /* CGLS or LSQR method to minimize the residuals*/
        
      ierr = KSPCreate(PETSC_COMM_WORLD,&ksp_min);CHKERRQ(ierr);
      if (tsirm->cgls) {
        ierr = KSPSetType(ksp_min,KSPCGLS);CHKERRQ(ierr);
      } else {
        ierr = KSPSetType(ksp_min,KSPLSQR);CHKERRQ(ierr);
      }
      ierr = KSPSetOperators(ksp_min,AS,AS);CHKERRQ(ierr);
      ierr = KSPSetTolerances(ksp_min,tsirm->tol_ls,PETSC_DEFAULT,PETSC_DEFAULT,tsirm->maxiter_ls);CHKERRQ(ierr);
      ierr = KSPGetPC(ksp_min,&pc_min);CHKERRQ(ierr);
      ierr = PCSetType(pc_min,PCNONE);CHKERRQ(ierr);
      ierr = KSPSolve(ksp_min,b,tsirm->Alpha);CHKERRQ(ierr);    /* Find Alpha such that ||AS Alpha = b|| */ 
      ierr = KSPGetIterationNumber(ksp_min,&iter_minimization);CHKERRQ(ierr);
      ierr = KSPGetResidualNorm(ksp_min,&norm);CHKERRQ(ierr);
      ierr = KSPMonitor(ksp_min,iter_minimization,norm);CHKERRQ(ierr);

      /* Apply minimization */
      ierr = MatMult(tsirm->S,tsirm->Alpha,x);CHKERRQ(ierr); /* x = S * Alpha */
    }
  } while (ksp->its<ksp->max_it && !ksp->reason);
  ierr = PetscFree(ind_row);CHKERRQ(ierr);
  ksp->its = total;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSetFromOptions_TSIRM"
PetscErrorCode KSPSetFromOptions_TSIRM(PetscOptionItems *PetscOptionsObject,KSP ksp)
{
  PetscErrorCode ierr;
  KSP_TSIRM      *tsirm = (KSP_TSIRM*)ksp->data;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"KSP TSIRM options");CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_tsirm_cgls","Method used for the minimization step","",tsirm->cgls,&tsirm->cgls,NULL);CHKERRQ(ierr); /*0:LSQR, 1:CGLS*/
  ierr = PetscOptionsReal("-ksp_tsirm_tol_ls","Tolerance threshold for the minimization step","",tsirm->tol_ls,&tsirm->tol_ls,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_tsirm_max_it_ls","Maximum number of iterations for the minimization step","",tsirm->maxiter_ls,&tsirm->maxiter_ls,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_tsirm_size_ls","Number of residuals for minimization","",tsirm->size_ls,&tsirm->size_ls,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPDestroy_TSIRM"
PetscErrorCode KSPDestroy_TSIRM(KSP ksp)
{
  KSP_TSIRM       *tsirm = (KSP_TSIRM*)ksp->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* Free work vectors */
  ierr = MatDestroy(&tsirm->S);CHKERRQ(ierr);
  ierr = VecDestroy(&tsirm->Alpha);CHKERRQ(ierr);
  ierr = VecDestroy(&tsirm->r);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPCreate_TSIRM"
PETSC_EXTERN PetscErrorCode KSPCreate_TSIRM(KSP ksp)
{
  PetscErrorCode ierr;
  KSP_TSIRM      *tsirm;
  
  PetscFunctionBegin;
  ierr                     = PetscNewLog(ksp,&tsirm);CHKERRQ(ierr);
  ksp->data                = (void*)tsirm;
  ierr                     = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr                     = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,1);CHKERRQ(ierr);
  ksp->ops->setup          = KSPSetUp_TSIRM;
  ksp->ops->solve          = KSPSolve_TSIRM;
  ksp->ops->destroy        = KSPDestroy_TSIRM;
  ksp->ops->buildsolution  = KSPBuildSolutionDefault;
  ksp->ops->buildresidual  = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = KSPSetFromOptions_TSIRM; 
  ksp->ops->view           = 0;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"This is not supported for complex numbers");
#endif
  PetscFunctionReturn(0);
}
