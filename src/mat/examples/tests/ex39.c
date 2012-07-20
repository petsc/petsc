
static char help[] = "Tests Elemental interface.\n\n"; 

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            Cdense,B,C,Ct;
  PetscInt       i,j,m = 5,n,nrows,ncols;
  const PetscInt *rows,*cols;
  IS             isrows,iscols;
  PetscErrorCode ierr;
  PetscScalar    *v;
  PetscMPIInt    rank,size;
  PetscReal      Cnorm;
  PetscBool      mats_view=PETSC_FALSE;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  n = m;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsHasName(PETSC_NULL,"-mats_view",&mats_view);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(C,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatGetOwnershipIS(C,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc(nrows*ncols*sizeof *v,&v);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      v[i*ncols+j] = (PetscReal)(10000*rank+100*rows[i]+cols[j]);
    }
  }
  ierr = MatSetValues(C,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);

  /* Test MatView(), MatDuplicate() and out-of-place MatConvert() */
  ierr = MatDuplicate(C,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  if (mats_view) {
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatConvert(C,MATMPIDENSE,MAT_INITIAL_MATRIX,&Cdense);CHKERRQ(ierr);
  if (mats_view) {
    ierr = MatView(Cdense,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  /* Test MatNorm() */
  ierr = MatNorm(C,NORM_1,&Cnorm);CHKERRQ(ierr);
  if (rank==0) printf("The 1-norm of C is %f\n",Cnorm);

  /* Test MatTranspose() and MatZeroEntries() */
  ierr = MatTranspose(C,MAT_INITIAL_MATRIX,&Ct);CHKERRQ(ierr);
  ierr = MatZeroEntries(Ct);CHKERRQ(ierr);

  /* Test MatAXPY(), MatAYPX() and in-place MatConvert() */
  ierr = MatCreate(PETSC_COMM_WORLD,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,m,n,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(B,MATELEMENTAL);CHKERRQ(ierr);
  ierr = MatSetFromOptions(B);CHKERRQ(ierr);
  ierr = MatSetUp(B);CHKERRQ(ierr);
  ierr = MatGetOwnershipIS(B,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      v[i*ncols+j] = (PetscReal)(1000*rows[i]+cols[j]);
    }
   }
  ierr = MatSetValues(B,nrows,rows,ncols,cols,v,INSERT_VALUES);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = ISRestoreIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISRestoreIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  if (mats_view) {
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatAXPY(B,2.5,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  if (mats_view) {
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = MatAYPX(B,3.75,C,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatConvert(B,MATDENSE,MAT_REUSE_MATRIX,&B);CHKERRQ(ierr);
  if (mats_view) {
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }
  ierr = ISDestroy(&isrows);CHKERRQ(ierr);
  ierr = ISDestroy(&iscols);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  
  /* Test MatMatTransposeMult() */
  ierr = MatMatTransposeMult(C,C,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&B);CHKERRQ(ierr);
  if (mats_view) {
    ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&Cdense);CHKERRQ(ierr);
  ierr = PetscFree(v);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&Ct);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}

 
