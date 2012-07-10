
static char help[] = "Tests LU, Cholesky factorization and MatMatSolve() for a Elemental dense matrix.\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A,F,B,X,C,Asym,G;
  Vec            b,x,c;
  PetscErrorCode ierr;
  PetscInt       m = 5,n,p,i,j,nrows,ncols;
  PetscScalar    *v,*vb,rval;
  PetscReal      norm,tol=1.e-15;
  PetscMPIInt    size,rank;
  PetscRandom    rand;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscBool      mats_view=PETSC_FALSE;
  MatFactorInfo  finfo;

  PetscInitialize(&argc,&argv,(char*) 0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);

  /* Get local dimensions of matrices */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  n = m;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  p = m/2;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&p,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-mats_view",&mats_view);CHKERRQ(ierr);

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
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = PetscMalloc(nrows*ncols*sizeof *v,&v);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      v[i*ncols+j] = rval;
      //v[i*ncols+j] = (PetscReal)(rank); 
      //v[i*ncols+j] = (PetscReal)(100*rows[i]+cols[j]+3.14); 
      //if (rank==-1) {ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] set (%d, %d, %g)\n",rank,rows[i],cols[j],v[i*ncols+j]);CHKERRQ(ierr);}
    }
  }
  ierr = MatSetValues(A,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  if (mats_view){
    Mat Aaij;
    ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatComputeExplicitOperator(A,&Aaij);CHKERRQ(ierr);
    ierr = MatView(Aaij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(&Aaij);CHKERRQ(ierr);
  }

  /* Create rhs matrix B */
  ierr = PetscPrintf(PETSC_COMM_WORLD," Create rhs matrix B ...\n");CHKERRQ(ierr);
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
  ierr = PetscMalloc(nrows*ncols*sizeof *v,&v);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
      v[i*ncols+j] = rval;
    }
  }
  ierr = MatSetValues(B,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  //ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  //ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  //ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  //ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  if (mats_view){
    Mat Baij;
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatComputeExplicitOperator(B,&Baij);CHKERRQ(ierr);
    ierr = MatView(Baij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(&Baij);CHKERRQ(ierr);
  }

  /* Create rhs vector b */
  ierr = VecCreate(PETSC_COMM_WORLD,&b);CHKERRQ(ierr);
  ierr = VecSetSizes(b,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(b);CHKERRQ(ierr);
  ierr = PetscMalloc(nrows*sizeof *vb,&vb);CHKERRQ(ierr);
  for (j=0; j<nrows; j++) {
    ierr = PetscRandomGetValue(rand,&rval);CHKERRQ(ierr);
    vb[j] = rval;
  }
  ierr = VecSetValues(b,nrows,rows,vb,INSERT_VALUES);CHKERRQ(ierr);
  ierr = PetscFree(vb);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = VecView(b,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);  


  /* Create X - same size as B */
  ierr = PetscPrintf(PETSC_COMM_WORLD," Create solution matrix X ...\n");CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = MatSetSizes(X,m,p,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(X,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(X);CHKERRQ(ierr);
  ierr = MatSetUp(X);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Create x - same size as b */
  ierr = VecDuplicate(b,&x);CHKERRQ(ierr);

  /* Cholesky factorization */
  ierr = PetscPrintf(PETSC_COMM_WORLD," Create Elemental matrix Asym\n");CHKERRQ(ierr);
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&Asym);CHKERRQ(ierr);
  ierr = MatAXPY(Asym,1.0,A,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  if (rank == 0) { 
    PetscInt M,N;
    ierr = MatGetSize(Asym,&M,&N);CHKERRQ(ierr);
    for (i=0; i<M; i++){
      rval = 100.0;
      ierr = MatSetValues(Asym,1,&i,1,&i,&rval,ADD_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(Asym,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Asym,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (mats_view){
    Mat Asymaij;
    ierr = MatView(Asym,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatComputeExplicitOperator(Asym,&Asymaij);CHKERRQ(ierr);
    ierr = MatView(Asymaij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(&Asymaij);CHKERRQ(ierr);
  }

  /*------------------------*/
  /* In-place Cholesky */
  /* Create matrix factor G, then copy Asym to G */
  ierr = MatCreate(PETSC_COMM_WORLD,&G);CHKERRQ(ierr);
  ierr = MatSetSizes(G,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(G,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(G);CHKERRQ(ierr);
  ierr = MatSetUp(G);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(G,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(G,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatCopy(Asym,G,SAME_NONZERO_PATTERN);CHKERRQ(ierr);

  /* Only G = U^T * U is implemented for now */ 
  finfo.dtcol = 0.1;
  ierr = MatCholeskyFactor(G,0,&finfo);CHKERRQ(ierr);
  if (mats_view){
    ierr = MatView(G,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Out-place Cholesky */
  ierr = MatDestroy(&G);CHKERRQ(ierr);
  

  /* LU factorization */
  /*------------------*/
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

  /* PF=LU or F=LU factorization - perms is ignored by Elemental; set finfo.dtcol !0 or 0 to enable/disable partial pivoting */ 
  finfo.dtcol = 0.1;
  ierr = MatLUFactor(F,0,0,&finfo);CHKERRQ(ierr);
  if (mats_view){
    ierr = MatView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Solve LUX = PB or LUX = B */
  ierr = MatSolve(F,b,x);CHKERRQ(ierr);
  ierr = MatMatSolve(F,B,X);CHKERRQ(ierr);
  //ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = MatView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);

  /* Check norm(A*X - B) */
  ierr = VecCreate(PETSC_COMM_WORLD,&c);CHKERRQ(ierr);
  ierr = VecSetSizes(c,m,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(c);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(c);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(c);CHKERRQ(ierr);
  ierr = MatMult(A,x,c);CHKERRQ(ierr);
  ierr = VecAXPY(c,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(c,NORM_1,&norm);CHKERRQ(ierr);
  //printf("vec norm is %f\n",norm);
  if (norm > tol){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for LU %G\n",norm);CHKERRQ(ierr);
  }

  ierr = MatMatMult(A,X,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&C);CHKERRQ(ierr);
  ierr = MatAXPY(C,-1.0,B,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatNorm(C,NORM_1,&norm);CHKERRQ(ierr);
  if (norm > tol){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for LU %G\n",norm);CHKERRQ(ierr);
  }

  /* Out-place LU */
  ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
  ierr = MatLUFactorSymbolic(F,A,0,0,&finfo);CHKERRQ(ierr);
  ierr = MatLUFactorNumeric(F,A,&finfo);CHKERRQ(ierr);
  // ierr = MatSolve(F,b,x);CHKERRQ(ierr);
  ierr = MatMatSolve(F,B,X);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);

  /* free space */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&Asym);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&c);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
