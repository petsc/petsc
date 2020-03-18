
static char help[] = "Tests LU, Cholesky factorization and MatMatSolve() for an Elemental dense matrix.\n\n";

#include <petscmat.h>

int main(int argc,char **argv)
{
  Mat            A,F,B,X,C,Aher,G;
  Vec            b,x,c,d,e;
  PetscErrorCode ierr;
  PetscInt       m = 5,n,p,i,j,nrows,ncols;
  PetscScalar    *v,*barray,rval;
  PetscReal      norm,tol=1.e-11;
  PetscMPIInt    size,rank;
  PetscRandom    rand;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscBool      mats_view=PETSC_FALSE;
  MatFactorInfo  finfo;

  ierr = PetscInitialize(&argc,&argv,(char*) 0,help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);

  /* Get local dimensions of matrices */
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  n    = m;
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  p    = m/2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-p",&p,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(NULL,NULL,"-mats_view",&mats_view);CHKERRQ(ierr);

  /* Create matrix A */
  ierr = PetscPrintf(PETSC_COMM_WORLD," Create Elemental matrix A\n");CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(A,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  /* Set local matrix entries */
  ierr = MatGetOwnershipIS(A,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrows*ncols,&v);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      ierr         = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      v[i*ncols+j] = rval;
    }
  }
  ierr = MatSetValues(A,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "A: nrows %d, m %d; ncols %d, n %d\n",nrows,m,ncols,n);CHKERRQ(ierr);
    ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Create rhs matrix B */
  ierr = PetscPrintf(PETSC_COMM_WORLD," Create rhs matrix B\n");CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,p,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(B,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatGetOwnershipIS(B,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrows*ncols,&v);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      ierr         = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      v[i*ncols+j] = rval;
    }
  }
  ierr = MatSetValues(B,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "B: nrows %d, m %d; ncols %d, p %d\n",nrows,m,ncols,p);CHKERRQ(ierr);
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Create rhs vector b and solution x (same size as b) */
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = VecGetArray(b,&barray);CHKERRQ(ierr);
  for (j=0; j<m; j++) {
    ierr      = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
    barray[j] = rval;
  }
  ierr = VecRestoreArray(b,&barray);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%d] b: m %d\n",rank,m);CHKERRQ(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT);CHKERRQ(ierr);
    ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  /* Create matrix X - same size as B */
  ierr = PetscPrintf(PETSC_COMM_WORLD," Create solution matrix X\n");CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = MatSetSizes(X,m,p,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(X,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(X);CHKERRQ(ierr);
  ierr = MatSetUp(X);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Cholesky factorization */
  /*------------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD," Create Elemental matrix Aher\n");CHKERRQ(ierr);
  ierr = MatHermitianTranspose(A,MAT_INITIAL_MATRIX,&Aher);CHKERRQ(ierr);
  ierr = MatAXPY(Aher,1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr); /* Aher = A + A^T */
  if(!rank) { /* add 100.0 to diagonals of Aher to make it spd */

    /* TODO: Replace this with a call to El::ShiftDiagonal( A, 100. ),
             or at least pre-allocate the right amount of space */
    PetscInt M,N;
    ierr = MatGetSize(Aher,&M,&N);CHKERRQ(ierr);
    for (i=0; i<M; i++) {
      rval = 100.0;
      ierr = MatSetValues(Aher,1,&i,1,&i,&rval,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Aher,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Aher,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Aher:\n");CHKERRQ(ierr);
    ierr = MatView(Aher,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Cholesky factorization */
  /*------------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD," Test Cholesky Solver \n");CHKERRQ(ierr);
  /* In-place Cholesky */
  /* Create matrix factor G, then copy Aher to G */
  ierr = MatCreate(PETSC_COMM_WORLD,&G);CHKERRQ(ierr);
  ierr = MatSetSizes(G,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(G,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(G);CHKERRQ(ierr);
  ierr = MatSetUp(G);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(G,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCopy(Aher,G,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  /* Only G = U^T * U is implemented for now */
  ierr = MatCholeskyFactor(G,0,0);CHKERRQ(ierr);
  if (mats_view) {
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Cholesky Factor G:\n");CHKERRQ(ierr);
    ierr = MatView(G,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Solve U^T * U x = b and U^T * U X = B */
  ierr = MatSolve(G,b,x);CHKERRQ(ierr);
  ierr = MatMatSolve(G,B,X);CHKERRQ(ierr);
  ierr = MatDestroy(&G);CHKERRQ(ierr);

  /* Out-place Cholesky */
  ierr = MatGetFactor(Aher,MATSOLVERELEMENTAL,MAT_FACTOR_CHOLESKY,&G);CHKERRQ(ierr);
  ierr = MatCholeskyFactorSymbolic(G,Aher,0,&finfo);CHKERRQ(ierr);
  ierr = MatCholeskyFactorNumeric(G,Aher,&finfo);CHKERRQ(ierr);
  if (mats_view) {
    ierr = MatView(G,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatSolve(G,b,x);CHKERRQ(ierr);
  ierr = MatMatSolve(G,B,X);CHKERRQ(ierr);
  ierr = MatDestroy(&G);CHKERRQ(ierr);

  /* Check norm(Aher*x - b) */
  ierr = VecCreate(PETSC_COMM_WORLD,&c);CHKERRQ(ierr);
  ierr = VecSetSizes(c,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(c);CHKERRQ(ierr);
  ierr = MatMult(Aher,x,c);CHKERRQ(ierr);
  ierr = VecAXPY(c,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(c,NORM_1,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: |Aher*x - b| for Cholesky %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* Check norm(Aher*X - B) */
  ierr = MatMatMult(Aher,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
  ierr = MatAXPY(C,-1.0,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(C,NORM_1,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: |Aher*X - B| for Cholesky %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* LU factorization */
  /*------------------*/
  ierr = PetscPrintf(PETSC_COMM_WORLD," Test LU Solver \n");CHKERRQ(ierr);
  /* In-place LU */
  /* Create matrix factor F, then copy A to F */
  ierr = MatCreate(PETSC_COMM_WORLD,&F);CHKERRQ(ierr);
  ierr = MatSetSizes(F,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(F,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(F);CHKERRQ(ierr);
  ierr = MatSetUp(F);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(F,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(F,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCopy(A,F,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  /* Create vector d to test MatSolveAdd() */
  ierr = VecDuplicate(x,&d);CHKERRQ(ierr);
  ierr = VecCopy(x,d);CHKERRQ(ierr);

  /* PF=LU or F=LU factorization - perms is ignored by Elemental;
     set finfo.dtcol !0 or 0 to enable/disable partial pivoting */
  finfo.dtcol = 0.1;
  ierr        = MatLUFactor(F,0,0,&finfo);CHKERRQ(ierr);

  /* Solve LUX = PB or LUX = B */
  ierr = MatSolveAdd(F,b,d,x);CHKERRQ(ierr);
  ierr = MatMatSolve(F,B,X);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);

  /* Check norm(A*X - B) */
  ierr = VecCreate(PETSC_COMM_WORLD,&e);CHKERRQ(ierr);
  ierr = VecSetSizes(e,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(e);CHKERRQ(ierr);
  ierr = MatMult(A,x,c);CHKERRQ(ierr);
  ierr = MatMult(A,d,e);CHKERRQ(ierr);
  ierr = VecAXPY(c,-1.0,e);CHKERRQ(ierr);
  ierr = VecAXPY(c,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(c,NORM_1,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: |A*x - b| for LU %g\n",(double)norm);CHKERRQ(ierr);
  }
  ierr = MatMatMult(A,X,MAT_REUSE_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
  ierr = MatAXPY(C,-1.0,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(C,NORM_1,&norm);CHKERRQ(ierr);
  if (norm > tol) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: |A*X - B| for LU %g\n",(double)norm);CHKERRQ(ierr);
  }

  /* Out-place LU */
  ierr = MatGetFactor(A,MATSOLVERELEMENTAL,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(F,A,0,0,&finfo);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(F,A,&finfo);CHKERRQ(ierr);
  if (mats_view) {
    ierr = MatView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatSolve(F,b,x);CHKERRQ(ierr);
  ierr = MatMatSolve(F,B,X);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);

  /* Free space */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Aher);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&c);CHKERRQ(ierr);
  ierr = VecDestroy(&d);CHKERRQ(ierr);
  ierr = VecDestroy(&e);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
