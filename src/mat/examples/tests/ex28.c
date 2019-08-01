static char help[] = "Illustrate how to do one symbolic factorization and multiple numeric factorizations using same matrix structure. \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscInt       ipack=0,i,rstart,rend,N=10,num_numfac=5,col[3],k;
  Mat            A[5],F;
  Vec            u,x,b;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscScalar    value[3];
  PetscReal      norm,tol=1.e-12;
  IS             perm,iperm;
  MatFactorInfo  info;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);

  /* Create and assemble matrices, all have same data structure */
  for (k=0; k<num_numfac; k++) {
    ierr = MatCreate(PETSC_COMM_WORLD,&A[k]);CHKERRQ(ierr);
    ierr = MatSetSizes(A[k],PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A[k]);CHKERRQ(ierr);
    ierr = MatSetUp(A[k]);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A[k],&rstart,&rend);CHKERRQ(ierr);

    value[0] = -1.0; value[1] = 2.0; value[2] = -1.0;
    for (i=rstart; i<rend; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      if (i == 0) {
        ierr = MatSetValues(A[k],1,&i,2,col+1,value+1,INSERT_VALUES);CHKERRQ(ierr);
      } else if (i == N-1) {
        ierr = MatSetValues(A[k],1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        ierr   = MatSetValues(A[k],1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(A[k],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A[k],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatSetOption(A[k],MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE);CHKERRQ(ierr);
  }

  /* Create vectors */
  ierr = MatCreateVecs(A[0],&x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&u);CHKERRQ(ierr);

  /* Set rhs vector b */
  ierr = VecSet(b,1.0);CHKERRQ(ierr);

  /* Get a symbolic factor F from A[0] */
  ierr = PetscOptionsGetInt(NULL,NULL,"-mat_solver_type",&ipack,NULL);CHKERRQ(ierr);
  switch (ipack) {
  case 1:
#if defined(PETSC_HAVE_SUPERLU)
    if (!rank) printf(" SUPERLU LU:\n");
    ierr = MatGetFactor(A[0],MATSOLVERSUPERLU,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    break;
#endif
  case 2:
#if defined(PETSC_HAVE_MUMPS)
    if (!rank) printf(" MUMPS LU:\n");
    ierr = MatGetFactor(A[0],MATSOLVERMUMPS,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
    {
      /* test mumps options */
      PetscInt icntl_7 = 5;
      ierr = MatMumpsSetIcntl(F,7,icntl_7);CHKERRQ(ierr);
    }
    break;
#endif
  default:
    if (!rank) printf(" PETSC LU:\n");
    ierr = MatGetFactor(A[0],MATSOLVERPETSC,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
  }

  info.fill = 5.0;
  ierr = MatGetOrdering(A[0],MATORDERINGNATURAL,&perm,&iperm);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(F,A[0],perm,iperm,&info);CHKERRQ(ierr);

  /* Compute numeric factors using same F, then solve */
  for (k = 0; k < num_numfac; k++) {
    /* Update A[k] */
    if (!rank) {
      value[0]=(PetscScalar)(k);
      i = 0;
      ierr = MatSetValues(A[k],1,&i,1,&i,value,ADD_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(A[k],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(A[k],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    /* Get numeric factor of A[k] */
    ierr = MatLUFactorNumeric(F,A[k],&info);CHKERRQ(ierr);

    /* Solve A[k] * x = b */
    ierr = MatSolve(F,b,x);CHKERRQ(ierr);

    /* Check the residual */
    ierr = MatMult(A[k],x,u);CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
    ierr = VecNorm(u,NORM_INFINITY,&norm);CHKERRQ(ierr);
    if (norm > tol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%d-the LU numfact and solve: residual %g\n",k,norm);CHKERRQ(ierr);
    }
  }

  /* Free data structures */
  for (k=0; k<num_numfac; k++) {
    ierr = MatDestroy(&A[k]);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = ISDestroy(&iperm);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -mat_solver_type 0
      requires: !single

   test:
      suffix: 2
      args: -mat_solver_type 1
      requires: superlu !single

   test:
      suffix: 3
      nsize: 2
      requires: mumps !single
      args: -mat_solver_type 2

TEST*/
