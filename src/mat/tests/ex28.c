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
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Create and assemble matrices, all have same data structure */
  for (k=0; k<num_numfac; k++) {
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A[k]));
    CHKERRQ(MatSetSizes(A[k],PETSC_DECIDE,PETSC_DECIDE,N,N));
    CHKERRQ(MatSetFromOptions(A[k]));
    CHKERRQ(MatSetUp(A[k]));
    CHKERRQ(MatGetOwnershipRange(A[k],&rstart,&rend));

    value[0] = -1.0*(k+1);
    value[1] =  2.0*(k+1);
    value[2] = -1.0*(k+1);
    for (i=rstart; i<rend; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      if (i == 0) {
        CHKERRQ(MatSetValues(A[k],1,&i,2,col+1,value+1,INSERT_VALUES));
      } else if (i == N-1) {
        CHKERRQ(MatSetValues(A[k],1,&i,2,col,value,INSERT_VALUES));
      } else {
        CHKERRQ(MatSetValues(A[k],1,&i,3,col,value,INSERT_VALUES));
      }
    }
    CHKERRQ(MatAssemblyBegin(A[k],MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(A[k],MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatSetOption(A[k],MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));
  }

  /* Create vectors */
  CHKERRQ(MatCreateVecs(A[0],&x,&b));
  CHKERRQ(VecDuplicate(x,&u));

  /* Set rhs vector b */
  CHKERRQ(VecSet(b,1.0));

  /* Get a symbolic factor F from A[0] */
  CHKERRQ(PetscStrncpy(solvertype,"petsc",sizeof(solvertype)));
  CHKERRQ(PetscOptionsGetString(NULL, NULL, "-mat_solver_type",solvertype,sizeof(solvertype),NULL));
  CHKERRQ(PetscOptionsGetEnum(NULL,NULL,"-mat_factor_type",MatFactorTypes,(PetscEnum*)&facttype,NULL));

  CHKERRQ(MatGetFactor(A[0],solvertype,facttype,&F));
  /* test mumps options */
#if defined(PETSC_HAVE_MUMPS)
  CHKERRQ(MatMumpsSetIcntl(F,7,5));
#endif
  CHKERRQ(PetscStrncpy(factortype,MatFactorTypes[facttype],sizeof(factortype)));
  CHKERRQ(PetscStrtoupper(solvertype));
  CHKERRQ(PetscStrtoupper(factortype));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," %s %s:\n",solvertype,factortype));

  CHKERRQ(MatFactorInfoInitialize(&info));
  info.fill = 5.0;
  CHKERRQ(MatGetOrdering(A[0],MATORDERINGNATURAL,&perm,&iperm));
  switch (facttype) {
  case MAT_FACTOR_LU:
    CHKERRQ(MatLUFactorSymbolic(F,A[0],perm,iperm,&info));
    break;
  case MAT_FACTOR_ILU:
    CHKERRQ(MatILUFactorSymbolic(F,A[0],perm,iperm,&info));
    break;
  case MAT_FACTOR_ICC:
    CHKERRQ(MatICCFactorSymbolic(F,A[0],perm,&info));
    break;
  case MAT_FACTOR_CHOLESKY:
    CHKERRQ(MatCholeskyFactorSymbolic(F,A[0],perm,&info));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not for factor type %s",factortype);
  }

  /* Compute numeric factors using same F, then solve */
  for (k = 0; k < num_numfac; k++) {
    switch (facttype) {
    case MAT_FACTOR_LU:
    case MAT_FACTOR_ILU:
      CHKERRQ(MatLUFactorNumeric(F,A[k],&info));
      break;
    case MAT_FACTOR_ICC:
    case MAT_FACTOR_CHOLESKY:
      CHKERRQ(MatCholeskyFactorNumeric(F,A[k],&info));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not for factor type %s",factortype);
    }

    /* Solve A[k] * x = b */
    CHKERRQ(MatSolve(F,b,x));

    /* Check the residual */
    CHKERRQ(MatMult(A[k],x,u));
    CHKERRQ(VecAXPY(u,-1.0,b));
    CHKERRQ(VecNorm(u,NORM_INFINITY,&norm));
    if (norm > tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT "-the %s numfact and solve: residual %g\n",k,factortype,(double)norm));
    }
  }

  /* Free data structures */
  for (k=0; k<num_numfac; k++) {
    CHKERRQ(MatDestroy(&A[k]));
  }
  CHKERRQ(MatDestroy(&F));
  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(ISDestroy(&iperm));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&u));
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
