
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

  CHKERRQ(PetscInitialize(&argc,&argv,(char*) 0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  CHKERRQ(PetscRandomCreate(PETSC_COMM_WORLD,&rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));

  /* Get local dimensions of matrices */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  n    = m;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  p    = m/2;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL));
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-mats_view",&mats_view));

  /* Create matrix A */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Create Elemental matrix A\n"));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,m,n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(A,MATELEMENTAL));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  /* Set local matrix entries */
  CHKERRQ(MatGetOwnershipIS(A,&isrows,&iscols));
  CHKERRQ(ISGetLocalSize(isrows,&nrows));
  CHKERRQ(ISGetIndices(isrows,&rows));
  CHKERRQ(ISGetLocalSize(iscols,&ncols));
  CHKERRQ(ISGetIndices(iscols,&cols));
  CHKERRQ(PetscMalloc1(nrows*ncols,&v));
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      CHKERRQ(PetscRandomGetValue(rand,&rval));
      v[i*ncols+j] = rval;
    }
  }
  CHKERRQ(MatSetValues(A,nrows,rows,ncols,cols,v,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(ISRestoreIndices(isrows,&rows));
  CHKERRQ(ISRestoreIndices(iscols,&cols));
  CHKERRQ(ISDestroy(&isrows));
  CHKERRQ(ISDestroy(&iscols));
  CHKERRQ(PetscFree(v));
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "A: nrows %" PetscInt_FMT ", m %" PetscInt_FMT "; ncols %" PetscInt_FMT ", n %" PetscInt_FMT "\n",nrows,m,ncols,n));
    CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Create rhs matrix B */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Create rhs matrix B\n"));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&B));
  CHKERRQ(MatSetSizes(B,m,p,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(B,MATELEMENTAL));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatGetOwnershipIS(B,&isrows,&iscols));
  CHKERRQ(ISGetLocalSize(isrows,&nrows));
  CHKERRQ(ISGetIndices(isrows,&rows));
  CHKERRQ(ISGetLocalSize(iscols,&ncols));
  CHKERRQ(ISGetIndices(iscols,&cols));
  CHKERRQ(PetscMalloc1(nrows*ncols,&v));
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      CHKERRQ(PetscRandomGetValue(rand,&rval));
      v[i*ncols+j] = rval;
    }
  }
  CHKERRQ(MatSetValues(B,nrows,rows,ncols,cols,v,INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY));
  CHKERRQ(ISRestoreIndices(isrows,&rows));
  CHKERRQ(ISRestoreIndices(iscols,&cols));
  CHKERRQ(ISDestroy(&isrows));
  CHKERRQ(ISDestroy(&iscols));
  CHKERRQ(PetscFree(v));
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "B: nrows %" PetscInt_FMT ", m %" PetscInt_FMT "; ncols %" PetscInt_FMT ", p %" PetscInt_FMT "\n",nrows,m,ncols,p));
    CHKERRQ(MatView(B,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Create rhs vector b and solution x (same size as b) */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&b));
  CHKERRQ(VecSetSizes(b,m,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(b));
  CHKERRQ(VecGetArray(b,&barray));
  for (j=0; j<m; j++) {
    CHKERRQ(PetscRandomGetValue(rand,&rval));
    barray[j] = rval;
  }
  CHKERRQ(VecRestoreArray(b,&barray));
  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));
  if (mats_view) {
    CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] b: m %" PetscInt_FMT "\n",rank,m));
    CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT));
    CHKERRQ(VecView(b,PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(VecDuplicate(b,&x));

  /* Create matrix X - same size as B */
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Create solution matrix X\n"));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&X));
  CHKERRQ(MatSetSizes(X,m,p,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(X,MATELEMENTAL));
  CHKERRQ(MatSetFromOptions(X));
  CHKERRQ(MatSetUp(X));
  CHKERRQ(MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY));

  /* Cholesky factorization */
  /*------------------------*/
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Create Elemental matrix Aher\n"));
  CHKERRQ(MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&Aher));
  CHKERRQ(MatAXPY(Aher,1.0,A,SAME_NONZERO_PATTERN)); /* Aher = A + A^T */
  if (rank == 0) { /* add 100.0 to diagonals of Aher to make it spd */

    /* TODO: Replace this with a call to El::ShiftDiagonal( A, 100.),
             or at least pre-allocate the right amount of space */
    PetscInt M,N;
    CHKERRQ(MatGetSize(Aher,&M,&N));
    for (i=0; i<M; i++) {
      rval = 100.0;
      CHKERRQ(MatSetValues(Aher,1,&i,1,&i,&rval,ADD_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(Aher,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Aher,MAT_FINAL_ASSEMBLY));
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Aher:\n"));
    CHKERRQ(MatView(Aher,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Cholesky factorization */
  /*------------------------*/
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Test Cholesky Solver \n"));
  /* In-place Cholesky */
  /* Create matrix factor G, then copy Aher to G */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&G));
  CHKERRQ(MatSetSizes(G,m,n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(G,MATELEMENTAL));
  CHKERRQ(MatSetFromOptions(G));
  CHKERRQ(MatSetUp(G));
  CHKERRQ(MatAssemblyBegin(G,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(G,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCopy(Aher,G,SAME_NONZERO_PATTERN));

  /* Only G = U^T * U is implemented for now */
  CHKERRQ(MatCholeskyFactor(G,0,0));
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Cholesky Factor G:\n"));
    CHKERRQ(MatView(G,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Solve U^T * U x = b and U^T * U X = B */
  CHKERRQ(MatSolve(G,b,x));
  CHKERRQ(MatMatSolve(G,B,X));
  CHKERRQ(MatDestroy(&G));

  /* Out-place Cholesky */
  CHKERRQ(MatGetFactor(Aher,MATSOLVERELEMENTAL,MAT_FACTOR_CHOLESKY,&G));
  CHKERRQ(MatCholeskyFactorSymbolic(G,Aher,0,&finfo));
  CHKERRQ(MatCholeskyFactorNumeric(G,Aher,&finfo));
  if (mats_view) {
    CHKERRQ(MatView(G,PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(MatSolve(G,b,x));
  CHKERRQ(MatMatSolve(G,B,X));
  CHKERRQ(MatDestroy(&G));

  /* Check norm(Aher*x - b) */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&c));
  CHKERRQ(VecSetSizes(c,m,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(c));
  CHKERRQ(MatMult(Aher,x,c));
  CHKERRQ(VecAXPY(c,-1.0,b));
  CHKERRQ(VecNorm(c,NORM_1,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: |Aher*x - b| for Cholesky %g\n",(double)norm));
  }

  /* Check norm(Aher*X - B) */
  CHKERRQ(MatMatMult(Aher,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C));
  CHKERRQ(MatAXPY(C,-1.0,B,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(C,NORM_1,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: |Aher*X - B| for Cholesky %g\n",(double)norm));
  }

  /* LU factorization */
  /*------------------*/
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Test LU Solver \n"));
  /* In-place LU */
  /* Create matrix factor F, then copy A to F */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&F));
  CHKERRQ(MatSetSizes(F,m,n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(F,MATELEMENTAL));
  CHKERRQ(MatSetFromOptions(F));
  CHKERRQ(MatSetUp(F));
  CHKERRQ(MatAssemblyBegin(F,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(F,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatCopy(A,F,SAME_NONZERO_PATTERN));
  /* Create vector d to test MatSolveAdd() */
  CHKERRQ(VecDuplicate(x,&d));
  CHKERRQ(VecCopy(x,d));

  /* PF=LU or F=LU factorization - perms is ignored by Elemental;
     set finfo.dtcol !0 or 0 to enable/disable partial pivoting */
  finfo.dtcol = 0.1;
  CHKERRQ(MatLUFactor(F,0,0,&finfo));

  /* Solve LUX = PB or LUX = B */
  CHKERRQ(MatSolveAdd(F,b,d,x));
  CHKERRQ(MatMatSolve(F,B,X));
  CHKERRQ(MatDestroy(&F));

  /* Check norm(A*X - B) */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&e));
  CHKERRQ(VecSetSizes(e,m,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(e));
  CHKERRQ(MatMult(A,x,c));
  CHKERRQ(MatMult(A,d,e));
  CHKERRQ(VecAXPY(c,-1.0,e));
  CHKERRQ(VecAXPY(c,-1.0,b));
  CHKERRQ(VecNorm(c,NORM_1,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: |A*x - b| for LU %g\n",(double)norm));
  }
  /* Reuse product C; replace Aher with A */
  CHKERRQ(MatProductReplaceMats(A,NULL,NULL,C));
  CHKERRQ(MatMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  CHKERRQ(MatAXPY(C,-1.0,B,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(C,NORM_1,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: |A*X - B| for LU %g\n",(double)norm));
  }

  /* Out-place LU */
  CHKERRQ(MatGetFactor(A,MATSOLVERELEMENTAL,MAT_FACTOR_LU,&F));
  CHKERRQ(MatLUFactorSymbolic(F,A,0,0,&finfo));
  CHKERRQ(MatLUFactorNumeric(F,A,&finfo));
  if (mats_view) {
    CHKERRQ(MatView(F,PETSC_VIEWER_STDOUT_WORLD));
  }
  CHKERRQ(MatSolve(F,b,x));
  CHKERRQ(MatMatSolve(F,B,X));
  CHKERRQ(MatDestroy(&F));

  /* Free space */
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&Aher));
  CHKERRQ(MatDestroy(&B));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&X));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&c));
  CHKERRQ(VecDestroy(&d));
  CHKERRQ(VecDestroy(&e));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(PetscRandomDestroy(&rand));
  CHKERRQ(PetscFinalize());
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
