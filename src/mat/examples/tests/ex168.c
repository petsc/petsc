
static char help[] = "Tests LU, Cholesky factorization and MatMatSolve() for a Elemental dense matrix.\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A,F,B,X;
  PetscErrorCode ierr;
  PetscInt       m = 5,n,p,i,j,nrows,ncols;
  PetscScalar    *v,rval;
  PetscReal      norm,tol=1.e-15;
  PetscMPIInt    size;
  PetscRandom    rand;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscBool      mats_view=PETSC_FALSE;
  MatFactorInfo  finfo;

  PetscInitialize(&argc,&argv,(char*) 0,help);
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
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  if (mats_view){
    Mat Baij;
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatComputeExplicitOperator(B,&Baij);CHKERRQ(ierr);
    ierr = MatView(Baij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = MatDestroy(&Baij);CHKERRQ(ierr);
  }

  /* Create X - same size as B */
  ierr = PetscPrintf(PETSC_COMM_WORLD," Create solution matrix X ...\n");CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = MatSetSizes(X,m,p,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(X,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(X);CHKERRQ(ierr);
  ierr = MatSetUp(X);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);


  /* Cholesky factorization */
  /*------------------------*/
  /* In-place Cholesky */
  /* Out-place Cholesky */
  

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
  // ierr = MatSolve(F,b,x);CHKERRQ(ierr);
  ierr = MatMatSolve(F,B,X);CHKERRQ(ierr);
  //ierr = MatView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);

  /* Check norm(A*X - B) */
  if(0) {
    if (norm > tol){
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for LU %G\n",norm);CHKERRQ(ierr);
    }
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
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
