
static char help[] = "Tests LU, Cholesky factorization and MatMatSolve() for an Elemental dense matrix.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,F,B,X,C,Aher,G;
  Vec            b,x,c,d,e;
  PetscInt       m = 5,n,p,i,j,nrows,ncols;
  PetscScalar    *v,*barray,rval;
  PetscReal      norm,tol=1.e-11;
  PetscMPIInt    size,rank;
  PetscRandom    rand;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscBool      mats_view=PETSC_FALSE;
  MatFactorInfo  finfo;

  PetscCall(PetscInitialize(&argc,&argv,(char*) 0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  PetscCall(PetscRandomSetFromOptions(rand));

  /* Get local dimensions of matrices */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  n    = m;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  p    = m/2;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL));
  PetscCall(PetscOptionsHasName(NULL,NULL,"-mats_view",&mats_view));

  /* Create matrix A */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Create Elemental matrix A\n"));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetSizes(A,m,n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(A,MATELEMENTAL));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  /* Set local matrix entries */
  PetscCall(MatGetOwnershipIS(A,&isrows,&iscols));
  PetscCall(ISGetLocalSize(isrows,&nrows));
  PetscCall(ISGetIndices(isrows,&rows));
  PetscCall(ISGetLocalSize(iscols,&ncols));
  PetscCall(ISGetIndices(iscols,&cols));
  PetscCall(PetscMalloc1(nrows*ncols,&v));
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      PetscCall(PetscRandomGetValue(rand,&rval));
      v[i*ncols+j] = rval;
    }
  }
  PetscCall(MatSetValues(A,nrows,rows,ncols,cols,v,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  PetscCall(ISRestoreIndices(isrows,&rows));
  PetscCall(ISRestoreIndices(iscols,&cols));
  PetscCall(ISDestroy(&isrows));
  PetscCall(ISDestroy(&iscols));
  PetscCall(PetscFree(v));
  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "A: nrows %" PetscInt_FMT ", m %" PetscInt_FMT "; ncols %" PetscInt_FMT ", n %" PetscInt_FMT "\n",nrows,m,ncols,n));
    PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Create rhs matrix B */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Create rhs matrix B\n"));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&B));
  PetscCall(MatSetSizes(B,m,p,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(B,MATELEMENTAL));
  PetscCall(MatSetFromOptions(B));
  PetscCall(MatSetUp(B));
  PetscCall(MatGetOwnershipIS(B,&isrows,&iscols));
  PetscCall(ISGetLocalSize(isrows,&nrows));
  PetscCall(ISGetIndices(isrows,&rows));
  PetscCall(ISGetLocalSize(iscols,&ncols));
  PetscCall(ISGetIndices(iscols,&cols));
  PetscCall(PetscMalloc1(nrows*ncols,&v));
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      PetscCall(PetscRandomGetValue(rand,&rval));
      v[i*ncols+j] = rval;
    }
  }
  PetscCall(MatSetValues(B,nrows,rows,ncols,cols,v,INSERT_VALUES));
  PetscCall(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  PetscCall(ISRestoreIndices(isrows,&rows));
  PetscCall(ISRestoreIndices(iscols,&cols));
  PetscCall(ISDestroy(&isrows));
  PetscCall(ISDestroy(&iscols));
  PetscCall(PetscFree(v));
  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "B: nrows %" PetscInt_FMT ", m %" PetscInt_FMT "; ncols %" PetscInt_FMT ", p %" PetscInt_FMT "\n",nrows,m,ncols,p));
    PetscCall(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Create rhs vector b and solution x (same size as b) */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&b));
  PetscCall(VecSetSizes(b,m,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(b));
  PetscCall(VecGetArray(b,&barray));
  for (j=0; j<m; j++) {
    PetscCall(PetscRandomGetValue(rand,&rval));
    barray[j] = rval;
  }
  PetscCall(VecRestoreArray(b,&barray));
  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));
  if (mats_view) {
    PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] b: m %" PetscInt_FMT "\n",rank,m));
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    PetscCall(VecView(b,PETSC_VIEWER_STDOUT_WORLD));
  }
  PetscCall(VecDuplicate(b,&x));

  /* Create matrix X - same size as B */
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Create solution matrix X\n"));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&X));
  PetscCall(MatSetSizes(X,m,p,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(X,MATELEMENTAL));
  PetscCall(MatSetFromOptions(X));
  PetscCall(MatSetUp(X));
  PetscCall(MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY));

  /* Cholesky factorization */
  /*------------------------*/
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Create Elemental matrix Aher\n"));
  PetscCall(MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&Aher));
  PetscCall(MatAXPY(Aher,1.0,A,SAME_NONZERO_PATTERN)); /* Aher = A + A^T */
  if (rank == 0) { /* add 100.0 to diagonals of Aher to make it spd */

    /* TODO: Replace this with a call to El::ShiftDiagonal( A, 100.),
             or at least pre-allocate the right amount of space */
    PetscInt M,N;
    PetscCall(MatGetSize(Aher,&M,&N));
    for (i=0; i<M; i++) {
      rval = 100.0;
      PetscCall(MatSetValues(Aher,1,&i,1,&i,&rval,ADD_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(Aher,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Aher,MAT_FINAL_ASSEMBLY));
  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Aher:\n"));
    PetscCall(MatView(Aher,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Cholesky factorization */
  /*------------------------*/
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Test Cholesky Solver \n"));
  /* In-place Cholesky */
  /* Create matrix factor G, then copy Aher to G */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&G));
  PetscCall(MatSetSizes(G,m,n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(G,MATELEMENTAL));
  PetscCall(MatSetFromOptions(G));
  PetscCall(MatSetUp(G));
  PetscCall(MatAssemblyBegin(G,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(G,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCopy(Aher,G,SAME_NONZERO_PATTERN));

  /* Only G = U^T * U is implemented for now */
  PetscCall(MatCholeskyFactor(G,0,0));
  if (mats_view) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Cholesky Factor G:\n"));
    PetscCall(MatView(G,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Solve U^T * U x = b and U^T * U X = B */
  PetscCall(MatSolve(G,b,x));
  PetscCall(MatMatSolve(G,B,X));
  PetscCall(MatDestroy(&G));

  /* Out-place Cholesky */
  PetscCall(MatGetFactor(Aher,MATSOLVERELEMENTAL,MAT_FACTOR_CHOLESKY,&G));
  PetscCall(MatCholeskyFactorSymbolic(G,Aher,0,&finfo));
  PetscCall(MatCholeskyFactorNumeric(G,Aher,&finfo));
  if (mats_view) PetscCall(MatView(G,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatSolve(G,b,x));
  PetscCall(MatMatSolve(G,B,X));
  PetscCall(MatDestroy(&G));

  /* Check norm(Aher*x - b) */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&c));
  PetscCall(VecSetSizes(c,m,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(c));
  PetscCall(MatMult(Aher,x,c));
  PetscCall(VecAXPY(c,-1.0,b));
  PetscCall(VecNorm(c,NORM_1,&norm));
  if (norm > tol) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Warning: |Aher*x - b| for Cholesky %g\n",(double)norm));
  }

  /* Check norm(Aher*X - B) */
  PetscCall(MatMatMult(Aher,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C));
  PetscCall(MatAXPY(C,-1.0,B,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(C,NORM_1,&norm));
  if (norm > tol) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Warning: |Aher*X - B| for Cholesky %g\n",(double)norm));
  }

  /* LU factorization */
  /*------------------*/
  PetscCall(PetscPrintf(PETSC_COMM_WORLD," Test LU Solver \n"));
  /* In-place LU */
  /* Create matrix factor F, then copy A to F */
  PetscCall(MatCreate(PETSC_COMM_WORLD,&F));
  PetscCall(MatSetSizes(F,m,n,PETSC_DECIDE,PETSC_DECIDE));
  PetscCall(MatSetType(F,MATELEMENTAL));
  PetscCall(MatSetFromOptions(F));
  PetscCall(MatSetUp(F));
  PetscCall(MatAssemblyBegin(F,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(F,MAT_FINAL_ASSEMBLY));
  PetscCall(MatCopy(A,F,SAME_NONZERO_PATTERN));
  /* Create vector d to test MatSolveAdd() */
  PetscCall(VecDuplicate(x,&d));
  PetscCall(VecCopy(x,d));

  /* PF=LU or F=LU factorization - perms is ignored by Elemental;
     set finfo.dtcol !0 or 0 to enable/disable partial pivoting */
  finfo.dtcol = 0.1;
  PetscCall(MatLUFactor(F,0,0,&finfo));

  /* Solve LUX = PB or LUX = B */
  PetscCall(MatSolveAdd(F,b,d,x));
  PetscCall(MatMatSolve(F,B,X));
  PetscCall(MatDestroy(&F));

  /* Check norm(A*X - B) */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&e));
  PetscCall(VecSetSizes(e,m,PETSC_DECIDE));
  PetscCall(VecSetFromOptions(e));
  PetscCall(MatMult(A,x,c));
  PetscCall(MatMult(A,d,e));
  PetscCall(VecAXPY(c,-1.0,e));
  PetscCall(VecAXPY(c,-1.0,b));
  PetscCall(VecNorm(c,NORM_1,&norm));
  if (norm > tol) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Warning: |A*x - b| for LU %g\n",(double)norm));
  }
  /* Reuse product C; replace Aher with A */
  PetscCall(MatProductReplaceMats(A,NULL,NULL,C));
  PetscCall(MatMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  PetscCall(MatAXPY(C,-1.0,B,SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(C,NORM_1,&norm));
  if (norm > tol) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Warning: |A*X - B| for LU %g\n",(double)norm));
  }

  /* Out-place LU */
  PetscCall(MatGetFactor(A,MATSOLVERELEMENTAL,MAT_FACTOR_LU,&F));
  PetscCall(MatLUFactorSymbolic(F,A,0,0,&finfo));
  PetscCall(MatLUFactorNumeric(F,A,&finfo));
  if (mats_view) PetscCall(MatView(F,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatSolve(F,b,x));
  PetscCall(MatMatSolve(F,B,X));
  PetscCall(MatDestroy(&F));

  /* Free space */
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Aher));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&X));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&c));
  PetscCall(VecDestroy(&d));
  PetscCall(VecDestroy(&e));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: elemental

   test:
      nsize: 2
      output_file: output/ex145.out

   test:
      suffix: 2
      nsize: 6
      output_file: output/ex145.out

TEST*/
