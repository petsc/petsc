
static char help[] = "Tests MatMatSolve() and MatMatTransposeSolve() for computing inv(A) with MUMPS.\n\
Example: mpiexec -n <np> ./ex214 -displ \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
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

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

#if !defined(PETSC_HAVE_MUMPS)
  if (rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF,"This example requires MUMPS, exit...\n"));
  PetscCall(PetscFinalize());
  return 0;
#else

  PetscCall(PetscOptionsGetBool(NULL,NULL,"-displ",&displ,NULL));

  /* Create matrix A */
  m = 4; n = 4;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A,5,NULL,5,NULL));
  PetscCall(MatSeqAIJSetPreallocation(A,5,NULL));

  PetscCall(MatGetOwnershipRange(A,&Istart,&Iend));
  for (Ii=Istart; Ii<Iend; Ii++) {
    v = -1.0; i = Ii/n; j = Ii - i*n;
    if (i>0)   {J = Ii - n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (i<m-1) {J = Ii + n; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j>0)   {J = Ii - 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
    if (j<n-1) {J = Ii + 1; PetscCall(MatSetValues(A,1,&Ii,1,&J,&v,ADD_VALUES));}
    v = 4.0; PetscCall(MatSetValues(A,1,&Ii,1,&Ii,&v,ADD_VALUES));
  }
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));

  PetscCall(MatGetLocalSize(A,&m,&n));
  PetscCall(MatGetSize(A,&M,&N));
  PetscCheckFalse(m != n,PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")", m, n);

  /* Create dense matrix C and X; C holds true solution with identical columns */
  nrhs = N;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-nrhs",&nrhs,NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&C));
  PetscCall(MatSetSizes(C,m,PETSC_DECIDE,PETSC_DECIDE,nrhs));
  PetscCall(MatSetType(C,MATDENSE));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(MatSetRandom(C,rand));
  PetscCall(MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&X));

  PetscCall(PetscStrcpy(solver,MATSOLVERMUMPS));
  if (rank == 0 && displ) PetscCall(PetscPrintf(PETSC_COMM_SELF,"Solving with %s: nrhs %" PetscInt_FMT ", size mat %" PetscInt_FMT " x %" PetscInt_FMT "\n",solver,nrhs,M,N));

  for (test=0; test<2; test++) {
    if (test == 0) {
      /* Test LU Factorization */
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"test LU factorization\n"));
      PetscCall(MatGetFactor(A,solver,MAT_FACTOR_LU,&F));
      PetscCall(MatLUFactorSymbolic(F,A,NULL,NULL,NULL));
      PetscCall(MatLUFactorNumeric(F,A,NULL));
    } else {
      /* Test Cholesky Factorization */
      PetscBool flg;
      PetscCall(MatIsSymmetric(A,0.0,&flg));
      PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"A must be symmetric for Cholesky factorization");

      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"test Cholesky factorization\n"));
      PetscCall(MatGetFactor(A,solver,MAT_FACTOR_CHOLESKY,&F));
      PetscCall(MatCholeskyFactorSymbolic(F,A,NULL,NULL));
      PetscCall(MatCholeskyFactorNumeric(F,A,NULL));
    }

    /* (1) Test MatMatSolve(): dense RHS = A*C, C: true solutions */
    /* ---------------------------------------------------------- */
    PetscCall(MatMatMult(A,C,MAT_INITIAL_MATRIX,2.0,&RHS));
    PetscCall(MatMatSolve(F,RHS,X));
    /* Check the error */
    PetscCall(MatAXPY(X,-1.0,C,SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(X,NORM_FROBENIUS,&norm));
    if (norm > tol) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"(1) MatMatSolve: Norm of error %g\n",norm));
    }

    /* Test X=RHS */
    PetscCall(MatMatSolve(F,RHS,RHS));
    /* Check the error */
    PetscCall(MatAXPY(RHS,-1.0,C,SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(RHS,NORM_FROBENIUS,&norm));
    if (norm > tol) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"(1.1) MatMatSolve: Norm of error %g\n",norm));
    }

    /* (2) Test MatMatSolve() for inv(A) with dense RHS:
     RHS = [e[0],...,e[nrhs-1]], dense X holds first nrhs columns of inv(A) */
    /* -------------------------------------------------------------------- */
    PetscCall(MatZeroEntries(RHS));
    for (i=0; i<nrhs; i++) {
      v = 1.0;
      PetscCall(MatSetValues(RHS,1,&i,1,&i,&v,INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(RHS,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(RHS,MAT_FINAL_ASSEMBLY));

    PetscCall(MatMatSolve(F,RHS,X));
    if (displ) {
      if (rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF," \n(2) first %" PetscInt_FMT " columns of inv(A) with dense RHS:\n",nrhs));
      PetscCall(MatView(X,PETSC_VIEWER_STDOUT_WORLD));
    }

    /* Check the residual */
    PetscCall(MatMatMult(A,X,MAT_INITIAL_MATRIX,2.0,&AX));
    PetscCall(MatAXPY(AX,-1.0,RHS,SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(AX,NORM_INFINITY,&norm));
    if (norm > tol) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"(2) MatMatSolve: Norm of residual %g\n",norm));
    }
    PetscCall(MatZeroEntries(X));

    /* (3) Test MatMatTransposeSolve() for inv(A) with sparse RHS stored in the host:
     spRHST = [e[0],...,e[nrhs-1]]^T, dense X holds first nrhs columns of inv(A) */
    /* --------------------------------------------------------------------------- */
    /* Create spRHST: PETSc does not support compressed column format which is required by MUMPS for sparse RHS matrix,
     thus user must create spRHST=spRHS^T and call MatMatTransposeSolve() */
    PetscCall(MatCreate(PETSC_COMM_WORLD,&spRHST));
    if (rank == 0) {
      /* MUMPS requirs RHS be centralized on the host! */
      PetscCall(MatSetSizes(spRHST,nrhs,M,PETSC_DECIDE,PETSC_DECIDE));
    } else {
      PetscCall(MatSetSizes(spRHST,0,0,PETSC_DECIDE,PETSC_DECIDE));
    }
    PetscCall(MatSetType(spRHST,MATAIJ));
    PetscCall(MatSetFromOptions(spRHST));
    PetscCall(MatSetUp(spRHST));
    if (rank == 0) {
      v = 1.0;
      for (i=0; i<nrhs; i++) {
        PetscCall(MatSetValues(spRHST,1,&i,1,&i,&v,INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(spRHST,MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(spRHST,MAT_FINAL_ASSEMBLY));

    PetscCall(MatMatTransposeSolve(F,spRHST,X));

    if (displ) {
      if (rank == 0) PetscCall(PetscPrintf(PETSC_COMM_SELF," \n(3) first %" PetscInt_FMT " columns of inv(A) with sparse RHS:\n",nrhs));
      PetscCall(MatView(X,PETSC_VIEWER_STDOUT_WORLD));
    }

    /* Check the residual: R = A*X - RHS */
    PetscCall(MatMatMult(A,X,MAT_REUSE_MATRIX,2.0,&AX));

    PetscCall(MatAXPY(AX,-1.0,RHS,SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(AX,NORM_INFINITY,&norm));
    if (norm > tol) {
      PetscCall(PetscPrintf(PETSC_COMM_SELF,"(3) MatMatSolve: Norm of residual %g\n",norm));
    }

    /* (4) Test MatMatSolve() for inv(A) with selected entries:
     input: spRHS gives selected indices; output: spRHS holds selected entries of inv(A) */
    /* --------------------------------------------------------------------------------- */
    if (nrhs == N) { /* mumps requires nrhs = n */
      /* Create spRHS on proc[0] */
      Mat spRHS = NULL;

      /* Create spRHS = spRHST^T. Two matrices share internal matrix data structure */
      PetscCall(MatCreateTranspose(spRHST,&spRHS));
      PetscCall(MatMumpsGetInverse(F,spRHS));
      PetscCall(MatDestroy(&spRHS));

      PetscCall(MatMumpsGetInverseTranspose(F,spRHST));
      if (displ) {
        PetscCall(PetscPrintf(PETSC_COMM_WORLD,"\nSelected entries of inv(A^T):\n"));
        PetscCall(MatView(spRHST,PETSC_VIEWER_STDOUT_WORLD));
      }
      PetscCall(MatDestroy(&spRHS));
    }
    PetscCall(MatDestroy(&AX));
    PetscCall(MatDestroy(&F));
    PetscCall(MatDestroy(&RHS));
    PetscCall(MatDestroy(&spRHST));
  }

  /* Free data structures */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&X));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
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
