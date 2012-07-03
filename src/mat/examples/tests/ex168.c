
static char help[] = "Tests LU, Cholesky factorization and MatMatSolve() for a Elemental dense matrix.\n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A,F,B,X;
  MatInfo        info;
  PetscErrorCode ierr;
  PetscInt       m = 5,n,i,j,nrows,ncols;
  PetscScalar    value = 1.0,*v;
  PetscReal      norm,tol=1.e-15;
  PetscMPIInt    size,rank;
  PetscRandom    rand;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscBool      flg;

  PetscInitialize(&argc,&argv,(char*) 0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  //if (size <= 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is parallel example only!");

  /* create matrix A */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  n = m;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);

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
      v[i*ncols+j] = (PetscReal)(rank*10000+100*rows[i]+cols[j]); 
      if (rank==-1) {ierr = PetscPrintf(PETSC_COMM_SELF,"[%d] set (%d, %d, %g)\n",rank,rows[i],cols[j],v[i*ncols+j]);CHKERRQ(ierr);}
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

  /* create rhs matrix B */

  
  /* Cholesky factorization - perm and factinfo are ignored by LAPACK */
  /* in-place Cholesky */
  /* out-place Cholesky */
  

  /* LU factorization - perms and factinfo are ignored by Elemental */
  /* in-place LU */
  //ierr = MatDuplicate(A,MAT_COPY_VALUES,&F);CHKERRQ(ierr);
  F = A;
  ierr = MatLUFactor(F,0,0,0);CHKERRQ(ierr);
  X = B; 
  //ierr = MatDuplicate(B,MAT_COPY_VALUES,&X);CHKERRQ(ierr);
  //ierr = MatMatSolve(F,X,X);CHKERRQ(ierr); 
#if defined(TMP)
  /* Check norm(A*X - B) */
  if (norm > tol){
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Warning: Norm of error for LU %G\n",norm);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&F);CHKERRQ(ierr);
#endif
  /* out-place LU */
  

  /* free space */
  //ierr = MatDestroy(&F);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  //ierr = MatDestroy(&B);CHKERRQ(ierr);
  //ierr = MatDestroy(&X);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
