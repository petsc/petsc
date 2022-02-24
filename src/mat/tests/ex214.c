
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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

#if !defined(PETSC_HAVE_MUMPS)
  if (rank == 0) CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"This example requires MUMPS, exit...\n"));
  ierr = PetscFinalize();
  return ierr;
#else

  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-displ",&displ,NULL));

  /* Create matrix A */
  m = 4; n = 4;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatMPIAIJSetPreallocation(A,5,NULL,5,NULL));
  CHKERRQ(MatSeqAIJSetPreallocation(A,5,NULL));

  CHKERRQ(MatGetOwnershipRange(A,&Istart,&Iend));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
    v = 4.0; CHKERRQ(MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatGetLocalSize(A,&m,&n));
  CHKERRQ(MatGetSize(A,&M,&N));
  PetscCheckFalse(m != n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")", m, n);

  /* Create dense matrix C and X; C holds true solution with identical columns */
  nrhs = N;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,m,PETSC_DECIDE,PETSC_DECIDE,nrhs));
  CHKERRQ(MatSetType(C,MATDENSE));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  CHKERRQ(MatSetRandom(C,rand));
  CHKERRQ(MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&X));

  CHKERRQ(PetscStrcpy(solver,MATSOLVERMUMPS));
  if (rank == 0 && displ) CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Solving with %s: nrhs %" PetscInt_FMT ", size mat %" PetscInt_FMT " x %" PetscInt_FMT "\n",solver,nrhs,M,N));

  for (test=0; test<2; test++) {
    if (test == 0) {
      /* Test LU Factorization */
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"test LU factorization\n"));
      CHKERRQ(MatGetFactor(A,solver,MAT_FACTOR_LU,&F));
      CHKERRQ(MatLUFactorSymbolic(F,A,NULL,NULL,NULL));
      CHKERRQ(MatLUFactorNumeric(F,A,NULL));
    } else {
      /* Test Cholesky Factorization */
      PetscBool flg;
      CHKERRQ(MatIsSymmetric(A,0.0,&flg));
      PetscCheckFalse(!flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"A must be symmetric for Cholesky factorization");

      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"test Cholesky factorization\n"));
      CHKERRQ(MatGetFactor(A,solver,MAT_FACTOR_CHOLESKY,&F));
      CHKERRQ(MatCholeskyFactorSymbolic(F,A,NULL,NULL));
      CHKERRQ(MatCholeskyFactorNumeric(F,A,NULL));
    }

    /* (1) Test MatMatSolve(): dense RHS = A*C, C: true solutions */
    /* ---------------------------------------------------------- */
    CHKERRQ(MatMatMult(A,C,MAT_INITIAL_MATRIX,2.0,&RHS));
    CHKERRQ(MatMatSolve(F,RHS,X));
    /* Check the error */
    CHKERRQ(MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(X,NORM_FROBENIUS,&norm));
    if (norm > tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"(1) MatMatSolve: Norm of error %g\n",norm));
    }

    /* Test X=RHS */
    CHKERRQ(MatMatSolve(F,RHS,RHS));
    /* Check the error */
    CHKERRQ(MatAXPY(RHS,-1.0,C,SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(RHS,NORM_FROBENIUS,&norm));
    if (norm > tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"(1.1) MatMatSolve: Norm of error %g\n",norm));
    }

    /* (2) Test MatMatSolve() for inv(A) with dense RHS:
     RHS = [e[0],...,e[nrhs-1]], dense X holds first nrhs columns of inv(A) */
    /* -------------------------------------------------------------------- */
    CHKERRQ(MatZeroEntries(RHS));
    for (i=0; i<nrhs; i++) {
      v = 1.0;
      CHKERRQ(MatSetValues(RHS,1,&i,1,&i,&v,INSERT_VALUES));
    }
    CHKERRQ(MatAssemblyBegin(RHS,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(RHS,MAT_FINAL_ASSEMBLY));

    CHKERRQ(MatMatSolve(F,RHS,X));
    if (displ) {
      if (rank == 0) CHKERRQ(PetscPrintf(PETSC_COMM_SELF," \n(2) first %" PetscInt_FMT " columns of inv(A) with dense RHS:\n",nrhs));
      CHKERRQ(MatView(X,PETSC_VIEWER_STDOUT_WORLD));
    }

    /* Check the residual */
    CHKERRQ(MatMatMult(A,X,MAT_INITIAL_MATRIX,2.0,&AX));
    CHKERRQ(MatAXPY(AX,-1.0,RHS,SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(AX,NORM_INFINITY,&norm));
    if (norm > tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"(2) MatMatSolve: Norm of residual %g\n",norm));
    }
    CHKERRQ(MatZeroEntries(X));

    /* (3) Test MatMatTransposeSolve() for inv(A) with sparse RHS stored in the host:
     spRHST = [e[0],...,e[nrhs-1]]^T, dense X holds first nrhs columns of inv(A) */
    /* --------------------------------------------------------------------------- */
    /* Create spRHST: PETSc does not support compressed column format which is required by MUMPS for sparse RHS matrix,
     thus user must create spRHST=spRHS^T and call MatMatTransposeSolve() */
    CHKERRQ(MatCreate(PETSC_COMM_WORLD,&spRHST));
    if (rank == 0) {
      /* MUMPS requirs RHS be centralized on the host! */
      CHKERRQ(MatSetSizes(spRHST,nrhs,M,PETSC_DECIDE,PETSC_DECIDE));
    } else {
      CHKERRQ(MatSetSizes(spRHST,0,0,PETSC_DECIDE,PETSC_DECIDE));
    }
    CHKERRQ(MatSetType(spRHST,MATAIJ));
    CHKERRQ(MatSetFromOptions(spRHST));
    CHKERRQ(MatSetUp(spRHST));
    if (rank == 0) {
      v = 1.0;
      for (i=0; i<nrhs; i++) {
        CHKERRQ(MatSetValues(spRHST,1,&i,1,&i,&v,INSERT_VALUES));
      }
    }
    CHKERRQ(MatAssemblyBegin(spRHST,MAT_FINAL_ASSEMBLY));
    CHKERRQ(MatAssemblyEnd(spRHST,MAT_FINAL_ASSEMBLY));

    CHKERRQ(MatMatTransposeSolve(F,spRHST,X));

    if (displ) {
      if (rank == 0) CHKERRQ(PetscPrintf(PETSC_COMM_SELF," \n(3) first %" PetscInt_FMT " columns of inv(A) with sparse RHS:\n",nrhs));
      CHKERRQ(MatView(X,PETSC_VIEWER_STDOUT_WORLD));
    }

    /* Check the residual: R = A*X - RHS */
    CHKERRQ(MatMatMult(A,X,MAT_REUSE_MATRIX,2.0,&AX));

    CHKERRQ(MatAXPY(AX,-1.0,RHS,SAME_NONZERO_PATTERN));
    CHKERRQ(MatNorm(AX,NORM_INFINITY,&norm));
    if (norm > tol) {
      CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"(3) MatMatSolve: Norm of residual %g\n",norm));
    }

    /* (4) Test MatMatSolve() for inv(A) with selected entries:
     input: spRHS gives selected indices; output: spRHS holds selected entries of inv(A) */
    /* --------------------------------------------------------------------------------- */
    if (nrhs == N) { /* mumps requires nrhs = n */
      /* Create spRHS on proc[0] */
      Mat spRHS = NULL;

      /* Create spRHS = spRHST^T. Two matrices share internal matrix data structure */
      CHKERRQ(MatCreateTranspose(spRHST,&spRHS));
      CHKERRQ(MatMumpsGetInverse(F,spRHS));
      CHKERRQ(MatDestroy(&spRHS));

      CHKERRQ(MatMumpsGetInverseTranspose(F,spRHST));
      if (displ) {
        CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"\nSelected entries of inv(A^T):\n"));
        CHKERRQ(MatView(spRHST,PETSC_VIEWER_STDOUT_WORLD));
      }
      CHKERRQ(MatDestroy(&spRHS));
    }
    CHKERRQ(MatDestroy(&AX));
    CHKERRQ(MatDestroy(&F));
    CHKERRQ(MatDestroy(&RHS));
    CHKERRQ(MatDestroy(&spRHST));
  }

  /* Free data structures */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&X));
  CHKERRQ(PetscRandomDestroy(&rand));
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
