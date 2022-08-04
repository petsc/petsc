static char help[] = "Illustrate how to do one symbolic factorization and multiple numeric factorizations using same matrix structure. \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscInt       i,rstart,rend,N=10,num_numfac=5,col[3],k;
  Mat            A[5],F;
  Vec            u,x,b;
  PetscMPIInt    rank;
  PetscScalar    value[3];
  PetscReal      norm,tol=100*PETSC_MACHINE_EPSILON;
  IS             perm,iperm;
  MatFactorInfo  info;
  MatFactorType  facttype = MAT_FACTOR_LU;
  char           solvertype[64];
  char           factortype[64];

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* Create and assemble matrices, all have same data structure */
  for (k=0; k<num_numfac; k++) {
    PetscCall(MatCreate(PETSC_COMM_WORLD,&A[k]));
    PetscCall(MatSetSizes(A[k],PETSC_DECIDE,PETSC_DECIDE,N,N));
    PetscCall(MatSetFromOptions(A[k]));
    PetscCall(MatSetUp(A[k]));
    PetscCall(MatGetOwnershipRange(A[k],&rstart,&rend));

    value[0] = -1.0*(k+1);
    value[1] =  2.0*(k+1);
    value[2] = -1.0*(k+1);
    for (i=rstart; i<rend; i++) {
      col[0] = i-1; col[1] = i; col[2] = i+1;
      if (i == 0) {
        PetscCall(MatSetValues(A[k],1,&i,2,col+1,value+1,INSERT_VALUES));
      } else if (i == N-1) {
        PetscCall(MatSetValues(A[k],1,&i,2,col,value,INSERT_VALUES));
      } else {
        PetscCall(MatSetValues(A[k],1,&i,3,col,value,INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(A[k],MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A[k],MAT_FINAL_ASSEMBLY));
    PetscCall(MatSetOption(A[k],MAT_NEW_NONZERO_LOCATION_ERR,PETSC_TRUE));
  }

  /* Create vectors */
  PetscCall(MatCreateVecs(A[0],&x,&b));
  PetscCall(VecDuplicate(x,&u));

  /* Set rhs vector b */
  PetscCall(VecSet(b,1.0));

  /* Get a symbolic factor F from A[0] */
  PetscCall(PetscStrncpy(solvertype,"petsc",sizeof(solvertype)));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-mat_solver_type",solvertype,sizeof(solvertype),NULL));
  PetscCall(PetscOptionsGetEnum(NULL,NULL,"-mat_factor_type",MatFactorTypes,(PetscEnum*)&facttype,NULL));

  PetscCall(MatGetFactor(A[0],solvertype,facttype,&F));
  /* test mumps options */
#if defined(PETSC_HAVE_MUMPS)
  PetscCall(MatMumpsSetIcntl(F,7,5));
#endif
  PetscCall(PetscStrncpy(factortype,MatFactorTypes[facttype],sizeof(factortype)));
  PetscCall(PetscStrtoupper(solvertype));
  PetscCall(PetscStrtoupper(factortype));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," %s %s:\n",solvertype,factortype));

  PetscCall(MatFactorInfoInitialize(&info));
  info.fill = 5.0;
  PetscCall(MatGetOrdering(A[0],MATORDERINGNATURAL,&perm,&iperm));
  switch (facttype) {
  case MAT_FACTOR_LU:
    PetscCall(MatLUFactorSymbolic(F,A[0],perm,iperm,&info));
    break;
  case MAT_FACTOR_ILU:
    PetscCall(MatILUFactorSymbolic(F,A[0],perm,iperm,&info));
    break;
  case MAT_FACTOR_ICC:
    PetscCall(MatICCFactorSymbolic(F,A[0],perm,&info));
    break;
  case MAT_FACTOR_CHOLESKY:
    PetscCall(MatCholeskyFactorSymbolic(F,A[0],perm,&info));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not for factor type %s",factortype);
  }

  /* Compute numeric factors using same F, then solve */
  for (k = 0; k < num_numfac; k++) {
    switch (facttype) {
    case MAT_FACTOR_LU:
    case MAT_FACTOR_ILU:
      PetscCall(MatLUFactorNumeric(F,A[k],&info));
      break;
    case MAT_FACTOR_ICC:
    case MAT_FACTOR_CHOLESKY:
      PetscCall(MatCholeskyFactorNumeric(F,A[k],&info));
      break;
    default:
      SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"Not for factor type %s",factortype);
    }

    /* Solve A[k] * x = b */
    PetscCall(MatSolve(F,b,x));

    /* Check the residual */
    PetscCall(MatMult(A[k],x,u));
    PetscCall(VecAXPY(u,-1.0,b));
    PetscCall(VecNorm(u,NORM_INFINITY,&norm));
    if (norm > tol) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"%" PetscInt_FMT "-the %s numfact and solve: residual %g\n",k,factortype,(double)norm));
    }
  }

  /* Free data structures */
  for (k=0; k<num_numfac; k++) {
    PetscCall(MatDestroy(&A[k]));
  }
  PetscCall(MatDestroy(&F));
  PetscCall(ISDestroy(&perm));
  PetscCall(ISDestroy(&iperm));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(PetscFinalize());
  return 0;
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
