static char help[] = "Illustrate how to do one symbolic factorization and multiple numeric factorizations using same matrix structure. \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscInt       i,rstart,rend,N=10,num_numfac=5,col[3],k;
  Mat            A[5],F;
  Vec            u,x,b;
  PetscErrorCode ierr;
  PetscMPIInt    rank;
  PetscScalar    value[3];
  PetscReal      norm,tol=100*PETSC_MACHINE_EPSILON;
  IS             perm,iperm;
  MatFactorInfo  info;
  MatFactorType  facttype = MAT_FACTOR_LU;
  char           solvertype[64];
  char           factortype[64];

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRMPI(ierr);

  /* Create and assemble matrices, all have same data structure */
  for (k=0; k<num_numfac; k++) {
    ierr = MatCreate(PETSC_COMM_WORLD,&A[k]);CHKERRQ(ierr);
    ierr = MatSetSizes(A[k],PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A[k]);CHKERRQ(ierr);
    ierr = MatSetUp(A[k]);CHKERRQ(ierr);
    ierr = MatGetOwnershipRange(A[k],&rstart,&rend);CHKERRQ(ierr);

    value[0] = -1.0*(k+1);
    value[1] =  2.0*(k+1);
    value[2] = -1.0*(k+1);
    for (i=rstart; i<rend; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      if (i == 0) {
        ierr = MatSetValues(A[k],1,&i,2,col+1,value+1,INSERT_VALUES);CHKERRQ(ierr);
      } else if (i == N-1) {
        ierr = MatSetValues(A[k],1,&i,2,col,value,INSERT_VALUES);CHKERRQ(ierr);
      } else {
        ierr = MatSetValues(A[k],1,&i,3,col,value,INSERT_VALUES);CHKERRQ(ierr);
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
  ierr = PetscStrncpy(solvertype,"petsc",sizeof(solvertype));CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL, NULL, "-mat_solver_type",solvertype,sizeof(solvertype),NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetEnum(NULL,NULL,"-mat_factor_type",MatFactorTypes,(PetscEnum*)&facttype,NULL);CHKERRQ(ierr);

  ierr = MatGetFactor(A[0],solvertype,facttype,&F);CHKERRQ(ierr);
  /* test mumps options */
#if defined(PETSC_HAVE_MUMPS)
  ierr = MatMumpsSetIcntl(F,7,5);CHKERRQ(ierr);
#endif
  ierr = PetscStrncpy(factortype,MatFactorTypes[facttype],sizeof(factortype));CHKERRQ(ierr);
  ierr = PetscStrtoupper(solvertype);CHKERRQ(ierr);
  ierr = PetscStrtoupper(factortype);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD," %s %s:\n",solvertype,factortype);CHKERRQ(ierr);

  ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
  info.fill = 5.0;
  ierr = MatGetOrdering(A[0],MATORDERINGNATURAL,&perm,&iperm);CHKERRQ(ierr);
  switch (facttype) {
  case MAT_FACTOR_LU:
    ierr = MatLUFactorSymbolic(F,A[0],perm,iperm,&info);CHKERRQ(ierr);
    break;
  case MAT_FACTOR_ILU:
    ierr = MatILUFactorSymbolic(F,A[0],perm,iperm,&info);CHKERRQ(ierr);
    break;
  case MAT_FACTOR_ICC:
    ierr = MatICCFactorSymbolic(F,A[0],perm,&info);CHKERRQ(ierr);
    break;
  case MAT_FACTOR_CHOLESKY:
    ierr = MatCholeskyFactorSymbolic(F,A[0],perm,&info);CHKERRQ(ierr);
    break;
  default:
    SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not for factor type %s\n",factortype);
  }

  /* Compute numeric factors using same F, then solve */
  for (k = 0; k < num_numfac; k++) {
    switch (facttype) {
    case MAT_FACTOR_LU:
    case MAT_FACTOR_ILU:
      ierr = MatLUFactorNumeric(F,A[k],&info);CHKERRQ(ierr);
      break;
    case MAT_FACTOR_ICC:
    case MAT_FACTOR_CHOLESKY:
      ierr = MatCholeskyFactorNumeric(F,A[k],&info);CHKERRQ(ierr);
      break;
    default:
      SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not for factor type %s\n",factortype);
    }

    /* Solve A[k] * x = b */
    ierr = MatSolve(F,b,x);CHKERRQ(ierr);

    /* Check the residual */
    ierr = MatMult(A[k],x,u);CHKERRQ(ierr);
    ierr = VecAXPY(u,-1.0,b);CHKERRQ(ierr);
    ierr = VecNorm(u,NORM_INFINITY,&norm);CHKERRQ(ierr);
    if (norm > tol) {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"%D-the %s numfact and solve: residual %g\n",k,factortype,(double)norm);CHKERRQ(ierr);
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

   test:
      suffix: 2
      args: -mat_solver_type superlu
      requires: superlu

   test:
      suffix: 3
      nsize: 2
      requires: mumps
      args: -mat_solver_type mumps

   test:
      suffix: 4
      args: -mat_solver_type cusparse -mat_type aijcusparse -mat_factor_type {{lu cholesky ilu icc}separate output}
      requires: cuda

TEST*/
