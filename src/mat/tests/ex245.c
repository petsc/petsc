
static char help[] = "Tests LU, Cholesky factorization and MatMatSolve() for a ScaLAPACK dense matrix.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,F,B,X,C,Aher,G;
  Vec            b,x,c,d,e;
  PetscErrorCode ierr;
  PetscInt       m=5,n,p,i,j,nrows,ncols;
  PetscScalar    *v,*barray,rval;
  PetscReal      norm,tol=1.e5*PETSC_MACHINE_EPSILON;
  PetscMPIInt    size,rank;
  PetscRandom    rand;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscBool      mats_view=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;
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
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Create ScaLAPACK matrix A\n"));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetSizes(A,m,n,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(A,MATSCALAPACK));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  /* Set local matrix entries */
  CHKERRQ(MatGetOwnershipIS(A,&isrows,&iscols));
  CHKERRQ(ISGetLocalSize(isrows,&nrows));
  CHKERRQ(ISGetIndices(isrows,&rows));
  CHKERRQ(ISGetLocalSize(iscols,&ncols));
  CHKERRQ(ISGetIndices(iscols,&cols));
  CHKERRQ(PetscMalloc1(nrows*ncols,&v));
  for (i=0;i<nrows;i++) {
    for (j=0;j<ncols;j++) {
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
  CHKERRQ(MatSetType(B,MATSCALAPACK));
  CHKERRQ(MatSetFromOptions(B));
  CHKERRQ(MatSetUp(B));
  CHKERRQ(MatGetOwnershipIS(B,&isrows,&iscols));
  CHKERRQ(ISGetLocalSize(isrows,&nrows));
  CHKERRQ(ISGetIndices(isrows,&rows));
  CHKERRQ(ISGetLocalSize(iscols,&ncols));
  CHKERRQ(ISGetIndices(iscols,&cols));
  CHKERRQ(PetscMalloc1(nrows*ncols,&v));
  for (i=0;i<nrows;i++) {
    for (j=0;j<ncols;j++) {
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
  for (j=0;j<m;j++) {
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
  CHKERRQ(MatDuplicate(B,MAT_DO_NOT_COPY_VALUES,&X));

  /* Cholesky factorization */
  /*------------------------*/
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Create ScaLAPACK matrix Aher\n"));
  CHKERRQ(MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&Aher));
  CHKERRQ(MatAXPY(Aher,1.0,A,SAME_NONZERO_PATTERN)); /* Aher = A + A^T */
  CHKERRQ(MatShift(Aher,100.0));  /* add 100.0 to diagonals of Aher to make it spd */
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Aher:\n"));
    CHKERRQ(MatView(Aher,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Cholesky factorization */
  /*------------------------*/
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Test Cholesky Solver \n"));
  /* In-place Cholesky */
  /* Create matrix factor G, with a copy of Aher */
  CHKERRQ(MatDuplicate(Aher,MAT_COPY_VALUES,&G));

  /* G = L * L^T */
  CHKERRQ(MatCholeskyFactor(G,0,0));
  if (mats_view) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Cholesky Factor G:\n"));
    CHKERRQ(MatView(G,PETSC_VIEWER_STDOUT_WORLD));
  }

  /* Solve L * L^T x = b and L * L^T * X = B */
  CHKERRQ(MatSolve(G,b,x));
  CHKERRQ(MatMatSolve(G,B,X));
  CHKERRQ(MatDestroy(&G));

  /* Out-place Cholesky */
  CHKERRQ(MatGetFactor(Aher,MATSOLVERSCALAPACK,MAT_FACTOR_CHOLESKY,&G));
  CHKERRQ(MatCholeskyFactorSymbolic(G,Aher,0,NULL));
  CHKERRQ(MatCholeskyFactorNumeric(G,Aher,NULL));
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
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: ||Aher*x - b||=%g for Cholesky\n",(double)norm));
  }

  /* Check norm(Aher*X - B) */
  CHKERRQ(MatMatMult(Aher,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C));
  CHKERRQ(MatAXPY(C,-1.0,B,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(C,NORM_1,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: ||Aher*X - B||=%g for Cholesky\n",(double)norm));
  }

  /* LU factorization */
  /*------------------*/
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD," Test LU Solver \n"));
  /* In-place LU */
  /* Create matrix factor F, with a copy of A */
  CHKERRQ(MatDuplicate(A,MAT_COPY_VALUES,&F));
  /* Create vector d to test MatSolveAdd() */
  CHKERRQ(VecDuplicate(x,&d));
  CHKERRQ(VecCopy(x,d));

  /* PF=LU factorization */
  CHKERRQ(MatLUFactor(F,0,0,NULL));

  /* Solve LUX = PB */
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
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: ||A*x - b||=%g for LU\n",(double)norm));
  }
  /* Reuse product C; replace Aher with A */
  CHKERRQ(MatProductReplaceMats(A,NULL,NULL,C));
  CHKERRQ(MatMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C));
  CHKERRQ(MatAXPY(C,-1.0,B,SAME_NONZERO_PATTERN));
  CHKERRQ(MatNorm(C,NORM_1,&norm));
  if (norm > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Warning: ||A*X - B||=%g for LU\n",(double)norm));
  }

  /* Out-place LU */
  CHKERRQ(MatGetFactor(A,MATSOLVERSCALAPACK,MAT_FACTOR_LU,&F));
  CHKERRQ(MatLUFactorSymbolic(F,A,0,0,NULL));
  CHKERRQ(MatLUFactorNumeric(F,A,NULL));
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
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   build:
      requires: scalapack

   test:
      nsize: 2
      output_file: output/ex245.out

   test:
      suffix: 2
      nsize: 6
      output_file: output/ex245.out

TEST*/
