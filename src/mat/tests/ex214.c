
static char help[] = "Tests MatMatSolve() and MatMatTransposeSolve() for computing inv(A) with MUMPS.\n\
Example: mpiexec -n <np> ./ex214 -displ \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
#if defined(PETSC_HAVE_MUMPS)
  Mat            A,RHS,C,F,X,AX,spRHST;
  PetscInt       m,n,nrhs,M,N,i,Istart,Iend,Ii,j,J,test;
  PetscScalar    v;
  PetscReal      norm,tol=PETSC_SQRT_MACHINE_EPSILON;
  PetscRandom    rand;
  PetscBool      displ=PETSC_FALSE;
  char           solver[256];
#endif

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

#if !defined(PETSC_HAVE_MUMPS)
  if (!rank) {ierr = PetscPrintf(PETSC_COMM_SELF,"This example requires MUMPS, exit...\n");CHKERRQ(ierr);}
  ierr = PetscFinalize();
  return ierr;
#else

  ierr = PetscOptionsGetBool(NULL,NULL,"-displ",&displ,NULL);CHKERRQ(ierr);

  /* Create matrix A */
  m = 4; n = 4;
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatMPIAIJSetPreallocation(A,5,NULL,5,NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(A,5,NULL);CHKERRQ(ierr);

  ierr = MatGetOwnershipRange(A,&Istart,&Iend);CHKERRQ(ierr);
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (i<m-1) {J = Ii + n; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j>0)   {J = Ii - 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    if (j<n-1) {J = Ii + 1; ierr = MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES);CHKERRQ(ierr);}
    v = 4.0; ierr = MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatGetLocalSize(A,&m,&n);CHKERRQ(ierr);
  ierr = MatGetSize(A,&M,&N);CHKERRQ(ierr);
  if (m != n) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%d, %d)", m, n);

  /* Create dense matrix C and X; C holds true solution with identical columns */
  nrhs = N;
  ierr = PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m,PETSC_DECIDE,PETSC_DECIDE,nrhs);CHKERRQ(ierr);
  ierr = MatSetType(C,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = MatSetRandom(C,rand);CHKERRQ(ierr);
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&X);CHKERRQ(ierr);

  ierr = PetscStrcpy(solver,MATSOLVERMUMPS);CHKERRQ(ierr);
  if (!rank && displ) {ierr = PetscPrintf(PETSC_COMM_SELF,"Solving with %s: nrhs %D, size mat %D x %D\n",solver,nrhs,M,N);CHKERRQ(ierr);}

  for (test=0; test<2; test++) {
    if (test == 0) {
      /* Test LU Factorization */
      ierr = PetscPrintf(PETSC_COMM_WORLD,"test LU factorization\n");CHKERRQ(ierr);
      ierr = MatGetFactor(A,solver,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
      ierr = MatLUFactorSymbolic(F,A,NULL,NULL,NULL);CHKERRQ(ierr);
      ierr = MatLUFactorNumeric(F,A,NULL);CHKERRQ(ierr);
    } else {
      /* Test Cholesky Factorization */
      PetscBool flg;
      ierr = MatIsSymmetric(A,0.0,&flg);CHKERRQ(ierr);
      if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"A must be symmetric for Cholesky factorization");

      ierr = PetscPrintf(PETSC_COMM_WORLD,"test Cholesky factorization\n");CHKERRQ(ierr);
      ierr = MatGetFactor(A,solver,MAT_FACTOR_CHOLESKY,&F);CHKERRQ(ierr);
      ierr = MatCholeskyFactorSymbolic(F,A,NULL,NULL);CHKERRQ(ierr);
      ierr = MatCholeskyFactorNumeric(F,A,NULL);CHKERRQ(ierr);
    }

    /* (1) Test MatMatSolve(): dense RHS = A*C, C: true solutions */
    /* ---------------------------------------------------------- */
    ierr = MatMatMult(A,C,MAT_INITIAL_MATRIX,2.0,&RHS);CHKERRQ(ierr);
    ierr = MatMatSolve(F,RHS,X);CHKERRQ(ierr);
    /* Check the error */
    ierr = MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(X,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
    if (norm > tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"(1) MatMatSolve: Norm of error %g\n",norm);CHKERRQ(ierr);
    }

    /* Test X=RHS */
    ierr = MatMatSolve(F,RHS,RHS);CHKERRQ(ierr);
    /* Check the error */
    ierr = MatAXPY(RHS,-1.0,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(RHS,NORM_FROBENIUS,&norm);CHKERRQ(ierr);
    if (norm > tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"(1.1) MatMatSolve: Norm of error %g\n",norm);CHKERRQ(ierr);
    }

    /* (2) Test MatMatSolve() for inv(A) with dense RHS:
     RHS = [e[0],...,e[nrhs-1]], dense X holds first nrhs columns of inv(A) */
    /* -------------------------------------------------------------------- */
    ierr = MatZeroEntries(RHS);CHKERRQ(ierr);
    for (i=0; i<nrhs; i++) {
      v = 1.0;
      ierr = MatSetValues(RHS,1,&i,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(RHS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(RHS,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = MatMatSolve(F,RHS,X);CHKERRQ(ierr);
    if (displ) {
      if (!rank) {ierr = PetscPrintf(PETSC_COMM_SELF," \n(2) first %D columns of inv(A) with dense RHS:\n",nrhs);CHKERRQ(ierr);}
      ierr = MatView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    /* Check the residual */
    ierr = MatMatMult(A,X,MAT_INITIAL_MATRIX,2.0,&AX);CHKERRQ(ierr);
    ierr = MatAXPY(AX,-1.0,RHS,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(AX,NORM_INFINITY,&norm);CHKERRQ(ierr);
    if (norm > tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"(2) MatMatSolve: Norm of residual %g\n",norm);CHKERRQ(ierr);
    }
    ierr = MatZeroEntries(X);CHKERRQ(ierr);

    /* (3) Test MatMatTransposeSolve() for inv(A) with sparse RHS stored in the host:
     spRHST = [e[0],...,e[nrhs-1]]^T, dense X holds first nrhs columns of inv(A) */
    /* --------------------------------------------------------------------------- */
    /* Create spRHST: PETSc does not support compressed column format which is required by MUMPS for sparse RHS matrix,
     thus user must create spRHST=spRHS^T and call MatMatTransposeSolve() */
    ierr = MatCreate(PETSC_COMM_WORLD,&spRHST);CHKERRQ(ierr);
    if (!rank) {
      /* MUMPS requirs RHS be centralized on the host! */
      ierr = MatSetSizes(spRHST,nrhs,M,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    } else {
      ierr = MatSetSizes(spRHST,0,0,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
    }
    ierr = MatSetType(spRHST,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetFromOptions(spRHST);CHKERRQ(ierr);
    ierr = MatSetUp(spRHST);CHKERRQ(ierr);
    if (!rank) {
      v = 1.0;
      for (i=0; i<nrhs; i++) {
        ierr = MatSetValues(spRHST,1,&i,1,&i,&v,INSERT_VALUES);CHKERRQ(ierr);
      }
    }
    ierr = MatAssemblyBegin(spRHST,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(spRHST,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

    ierr = MatMatTransposeSolve(F,spRHST,X);CHKERRQ(ierr);

    if (displ) {
      if (!rank) {ierr = PetscPrintf(PETSC_COMM_SELF," \n(3) first %D columns of inv(A) with sparse RHS:\n",nrhs);CHKERRQ(ierr);}
      ierr = MatView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    }

    /* Check the residual: R = A*X - RHS */
    ierr = MatMatMult(A,X,MAT_REUSE_MATRIX,2.0,&AX);CHKERRQ(ierr);

    ierr = MatAXPY(AX,-1.0,RHS,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = MatNorm(AX,NORM_INFINITY,&norm);CHKERRQ(ierr);
    if (norm > tol) {
      ierr = PetscPrintf(PETSC_COMM_SELF,"(3) MatMatSolve: Norm of residual %g\n",norm);CHKERRQ(ierr);
    }

    /* (4) Test MatMatSolve() for inv(A) with selected entries:
     input: spRHS gives selected indices; output: spRHS holds selected entries of inv(A) */
    /* --------------------------------------------------------------------------------- */
    if (nrhs == N) { /* mumps requires nrhs = n */
      /* Create spRHS on proc[0] */
      Mat spRHS = NULL;

      /* Create spRHS = spRHST^T. Two matrices share internal matrix data structure */
      ierr = MatCreateTranspose(spRHST,&spRHS);CHKERRQ(ierr);
      ierr = MatMumpsGetInverse(F,spRHS);CHKERRQ(ierr);
      ierr = MatDestroy(&spRHS);CHKERRQ(ierr);

      ierr = MatMumpsGetInverseTranspose(F,spRHST);CHKERRQ(ierr);
      if (displ) {
        ierr = PetscPrintf(PETSC_COMM_WORLD,"\nSelected entries of inv(A^T):\n");CHKERRQ(ierr);
        ierr = MatView(spRHST,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      }
      ierr = MatDestroy(&spRHS);CHKERRQ(ierr);
    }
    ierr = MatDestroy(&AX);CHKERRQ(ierr);
    ierr = MatDestroy(&F);CHKERRQ(ierr);
    ierr = MatDestroy(&RHS);CHKERRQ(ierr);
    ierr = MatDestroy(&spRHST);CHKERRQ(ierr);
  }

  /* Free data structures */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
#endif
}

/*TEST

   test:
     requires: mumps double !complex

   test:
     suffix: 2
     requires: mumps double !complex
     nsize: 2

TEST*/
