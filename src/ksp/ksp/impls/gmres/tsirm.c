/*
    This file implements TSIRM, the Two-Stage Iteration with least-squares Residual Minimization method. 
    It is an iterative method to solve large sparse linear systems fo the form Ax=b, and it improves the convergence of Krylov based iterative methods.
    The principle is to build an external iteration over a Krylov method (for example GMRES), and to frequently store its current residual in a matrix S. After a given number of outer iterations, a least-squares minimization step (with CGLS or LSQR) is applied on S, in order to compute a better solution and to make new iterations if required.
*/
#include <petsc/private/kspimpl.h>	/*I "petscksp.h" I*/
#include <petscksp.h>

#undef __FUNCT__
#define __FUNCT__ "KSPSetUp_TSIRM"
static PetscErrorCode KSPSetUp_TSIRM(KSP ksp)
{
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPSolve_TSIRM"
PetscErrorCode KSPSolve_TSIRM(KSP ksp)
{
  PetscErrorCode ierr;
  KSP            sub_ksp;
  PC             pc;
  PCType         type;
  Mat            S,AS,A;
  Vec            Alpha,x,b,r,Ax;
  PetscScalar    T1,T2,*array;
  PetscReal      tol_ls = 1e-40,norm = 20;
  PetscInt       size,Istart,Iend,i,*ind_row,first_iteration = 1,its = 0,total = 0,col = 0;
  PetscInt       size_ls = 12,iter_minimization = 0,maxiter_ls = 15,cgls = 0,restart = 30;
  
   PetscFunctionBegin;  
  PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"Tsirm","");
  ierr = PetscOptionsInt("-ksp_tsirm_cgls","Method used for the minimization step","",cgls,&cgls,NULL);CHKERRQ(ierr); /*0:LSQR, 1:CGLS*/
  ierr = PetscOptionsReal("-ksp_tsirm_tol_ls","Tolerance threshold for the minimization step","",tol_ls,&tol_ls,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_tsirm_maxiter_ls","Maximum number of iterations for the minimization step","",maxiter_ls,&maxiter_ls,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsInt("-ksp_tsirm_size_ls","Number of residuals for minimization","",size_ls,&size_ls,NULL);CHKERRQ(ierr);
  PetscOptionsEnd();
  
  /* System of equations */
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n***** Enter tsirm:\n");CHKERRQ(ierr);
  KSPGetRhs(ksp,&b);            /* Right-hand side vector */
  KSPGetOperators(ksp,&A,NULL); /* Matrix of the system   */
  KSPGetSolution(ksp,&x);       /* Solution vector        */
  VecGetSize(b,&size);          /* Size of the system     */
  MatGetOwnershipRange(A,&Istart,&Iend);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\tSize of vector %D\n",size);CHKERRQ(ierr);
  
  /* Matrix S of residuals */
  ierr = MatCreate(PETSC_COMM_WORLD,&S);CHKERRQ(ierr);
  ierr = MatSetSizes(S,Iend-Istart,PETSC_DECIDE,size,size_ls);CHKERRQ(ierr);
  ierr = MatSetType(S,MATMPIDENSE);CHKERRQ(ierr);
  ierr = MatSetUp(S);CHKERRQ(ierr);
  
  /* Vector Alpha computed in the minimization step */
  ierr = VecCreate(PETSC_COMM_WORLD,&Alpha);CHKERRQ(ierr);
  ierr = VecSetSizes(Alpha,PETSC_DECIDE,size_ls);CHKERRQ(ierr);
  ierr = VecSetFromOptions(Alpha);CHKERRQ(ierr);
  
  ierr = VecDuplicate(b,&r);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&Ax);CHKERRQ(ierr);
  
  /* Row indexes (these indexes are global) */
  ind_row = (PetscInt*)malloc(sizeof(PetscInt)*(Iend-Istart));
  for (i=0;i<Iend-Istart;i++) ind_row[i] = i+Istart;
  
  /* TSIRM code */
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);

  {
    PetscReal rtol,abstol,dtol;
    PetscInt maxits;
    ierr = KSPGetTolerances(ksp,&rtol,&abstol,&dtol,&maxits);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\tTSIRM: rtol %.2e, abstol %.2e, dtol %.2e, maxits %d\n",rtol,abstol,dtol,maxits);CHKERRQ(ierr);
  }
  
  {
    PetscReal rtol,abstol,dtol;
    PetscInt maxits;
    ierr = PCKSPGetKSP(pc,&sub_ksp);CHKERRQ(ierr);
    ierr = KSPGetType(sub_ksp,&type);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(sub_ksp);CHKERRQ(ierr);
    ierr = KSPGetTolerances(sub_ksp,&rtol,&abstol,&dtol,&maxits);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"\tInner Iteration: rtol %.2e, abstol %.2e, dtol %.2e, maxits %d\n",rtol,abstol,dtol,maxits);CHKERRQ(ierr);
  }
  
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\tKSP type of the inner iteration: %s\n",type);CHKERRQ(ierr);
  
  T1 = MPI_Wtime();
  /* previously it seemed good but with SNES it seems not good... */
  ierr = KSP_MatMult(sub_ksp,A,x,r);CHKERRQ(ierr);
  VecAXPY(r,-1,b);
  VecNorm(r,NORM_2,&norm);
  ksp->its = 0;
  KSPConvergedDefault(ksp,ksp->its,norm,&ksp->reason,ksp->cnvP);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\tConverged: %d, Residual norm: %.2e, Iterations: %d\n",ksp->reason,norm,ksp->its);CHKERRQ(ierr);
  ierr = KSPSetInitialGuessNonzero(sub_ksp,PETSC_TRUE);CHKERRQ(ierr);
  do {
    for (col=0;col<size_ls && ksp->reason==0;col++) {
      /* Solve (inner ietration) */
      ierr = KSPSolve(sub_ksp,b,x);CHKERRQ(ierr);
      ierr = KSPGetIterationNumber(sub_ksp,&its);CHKERRQ(ierr);
      total += its;
      
      /* Build S^T */
      ierr = VecGetArray(x,&array);
      ierr = MatSetValues(S,Iend-Istart,ind_row,1,&col,array,INSERT_VALUES);
      VecRestoreArray(x,&array);
      
      KSPGetResidualNorm(sub_ksp,&norm);
      ksp->rnorm = norm;
      ksp->its ++;
      KSPConvergedDefault(ksp,ksp->its,norm,&ksp->reason,ksp->cnvP);
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Converged: %D,	Residual norm: %.2g,	Outer iterations: %D,	Inner iteration: %D	Total iterations: %D\n",ksp->reason,ksp->rnorm,ksp->its,its,total);CHKERRQ(ierr);
    }
    
    /* Minimization step */
    if (!ksp->reason) {
      MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);
      MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);
      if (first_iteration) {
        MatMatMult(A,S,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&AS);
        first_iteration = 0;
      } else {
        MatMatMult(A,S,MAT_REUSE_MATRIX,PETSC_DEFAULT,&AS);
      }
      
      if (cgls) {
        /* CGLS method */
        KSP ksp3;
        PC pc;
        
        ierr = KSPCreate(PETSC_COMM_WORLD,&ksp3);CHKERRQ(ierr);
        ierr = KSPSetType(ksp3,KSPCGLS);CHKERRQ(ierr);
        ierr = KSPSetOperators(ksp3,AS,AS);CHKERRQ(ierr);
        ierr = KSPSetTolerances(ksp3,tol_ls,PETSC_DEFAULT,PETSC_DEFAULT,maxiter_ls);CHKERRQ(ierr);
        ierr = KSPGetPC(ksp3,&pc);CHKERRQ(ierr);
        ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
        ierr = KSPSolve(ksp3,b,Alpha);CHKERRQ(ierr);
        ierr = KSPGetIterationNumber(ksp3,&iter_minimization);CHKERRQ(ierr);
        ierr = KSPGetResidualNorm(ksp3,&norm);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nCGLS: iter %d, norm %.2e\n\n",iter_minimization,norm);CHKERRQ(ierr);
      } else {
        /* LSQR method */
        KSP ksp3;
        PC pc;
        
        ierr = KSPCreate(PETSC_COMM_WORLD,&ksp3);CHKERRQ(ierr);
        ierr = KSPSetType(ksp3,KSPLSQR);CHKERRQ(ierr);
        ierr = KSPSetOperators(ksp3,AS,AS);CHKERRQ(ierr);
        ierr = KSPSetTolerances(ksp3,tol_ls,PETSC_DEFAULT,PETSC_DEFAULT,maxiter_ls);CHKERRQ(ierr);
        ierr = KSPGetPC(ksp3,&pc);CHKERRQ(ierr);
        ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);
        ierr = KSPSolve(ksp3,b,Alpha);CHKERRQ(ierr);
        ierr = KSPGetIterationNumber(ksp3,&iter_minimization);CHKERRQ(ierr);
        ierr = KSPGetResidualNorm(ksp3,&norm);CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nLSQR: iter %d, norm %.2e\n\n",iter_minimization,norm);CHKERRQ(ierr);
      }
      
      /* Minimizer */
      MatMult(S,Alpha,x); /* x = S * Alpha */
    }
  } while (ksp->its<ksp->max_it && !ksp->reason);
  
  ksp->its = total;
  T2 = MPI_Wtime();
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n\t\t\t -- Execution time of the step			: %g (s)\n",T2-T1);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\t\t\t -- Total number of iterations of the step	: %D\n\n",total);CHKERRQ(ierr);
  
  ierr = VecDestroy(&Alpha);CHKERRQ(ierr);
  ierr = MatDestroy(&S);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = VecDestroy(&Ax);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "KSPCreate_TSIRM"
PETSC_EXTERN PetscErrorCode KSPCreate_TSIRM(KSP ksp)
{
  PetscErrorCode ierr;
  
  PetscFunctionBegin;
  ksp->data = (void*)0;
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_PRECONDITIONED,PC_LEFT,2);CHKERRQ(ierr);
  ierr = KSPSetSupportedNorm(ksp,KSP_NORM_UNPRECONDITIONED,PC_RIGHT,1);CHKERRQ(ierr);
  ksp->ops->setup = KSPSetUp_TSIRM;
  ksp->ops->solve = KSPSolve_TSIRM;
  ksp->ops->destroy = KSPDestroyDefault;
  ksp->ops->buildsolution = KSPBuildSolutionDefault;
  ksp->ops->buildresidual = KSPBuildResidualDefault;
  ksp->ops->setfromoptions = 0;
  ksp->ops->view = 0;
#if defined(PETSC_USE_COMPLEX)
  SETERRQ(PetscObjectComm((PetscObject)ksp),PETSC_ERR_SUP,"This is not supported for complex numbers");
#endif
  PetscFunctionReturn(0);
}
