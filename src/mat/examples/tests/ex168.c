
static char help[] = "Tests LU, Cholesky factorization and MatMatSolve() for a Elemental dense matrix.\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A,F,B,X,Aaij,Baij;
  MatInfo        info;
  PetscErrorCode ierr;
  PetscInt       m = 5,n,p,i,j,nrows,ncols;
  PetscScalar    value = 1.0,*v;
  PetscReal      norm,tol=1.e-15;
  PetscMPIInt    size,rank;
  PetscRandom    rand;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscBool      flg;
  MatFactorInfo  finfo;

  PetscInitialize(&argc,&argv,(char*) 0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  //if (size <= 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is parallel example only!");

  /* create matrix A */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  n = m;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  p = m;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-p",&p,PETSC_NULL);CHKERRQ(ierr);

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
      //v[i*ncols+j] = (PetscReal)(rank); 
      v[i*ncols+j] = (PetscReal)(100*rows[i]+cols[j]+3.14); 
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
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /*ierr = MatComputeExplicitOperator(A,&Aaij);CHKERRQ(ierr);
  ierr = MatView(Aaij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
   ierr = MatDestroy(&Aaij);CHKERRQ(ierr);*/

  /* create rhs matrix B */
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
      //v[i*ncols+j] = (PetscReal)(rank);
      v[i*ncols+j] = (PetscReal)(1000*rows[i]+cols[j]);
      //if (rank==-1) {ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] set (%d, %d, %g)\n",rank,rows[i],cols[j],v[i*ncols+j]);CHKERRQ(ierr);}
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
  ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  /*ierr = MatComputeExplicitOperator(B,&Baij);CHKERRQ(ierr);
  ierr = MatView(Baij,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
   ierr = MatDestroy(&Baij);CHKERRQ(ierr);*/


  /* Cholesky factorization - perm and factinfo are ignored by LAPACK */
  /* in-place Cholesky */
  /* out-place Cholesky */
  

  /* Create F - same size as A */
  ierr = MatCreate(PETSC_COMM_WORLD,&F);CHKERRQ(ierr);
  ierr = MatSetSizes(F,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(F,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(F);CHKERRQ(ierr);
  ierr = MatSetUp(F);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(F,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(F,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Copy A to F */
  ierr = MatCopy(A,F,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  /* PF=LU or F=LU factorization - perms is ignored by Elemental; set finfo.dtcol to 1 or 0 to enable/disable partial pivoting  */ 
  finfo.dtcol = 0;
  ierr = MatLUFactor(F,0,0,&finfo);CHKERRQ(ierr);
  ierr = MatView(F,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Create X - same size as B */
  ierr = MatCreate(PETSC_COMM_WORLD,&X);CHKERRQ(ierr);
  ierr = MatSetSizes(X,m,p,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(X,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(X);CHKERRQ(ierr);
  ierr = MatSetUp(X);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(X,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Solve LUX=PB or LUX=B */
  ierr = MatMatSolve(F,B,X);CHKERRQ(ierr);
  ierr = MatView(X,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);

  /* Check norm(A*X - B) */
  if(0) {
    if (norm > tol){
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for LU %G\n",norm);CHKERRQ(ierr);
    }
  }

  /* out-place LU */
  ierr = MatGetFactor(A,MATSOLVERPETSC,MAT_FACTOR_LU,&F);CHKERRQ(ierr);
  //ierr = MatLUFactorSymbolic(F,A,0,0,0);CHKERRQ(ierr);
  //ierr = MatLUFactorNumeric(F,A,0);CHKERRQ(ierr);
  //ierr = MatSolve(F,B,X);CHKERRQ(ierr);
  ierr = MatDestroy(&F);CHKERRQ(ierr);

  /* free space */
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
