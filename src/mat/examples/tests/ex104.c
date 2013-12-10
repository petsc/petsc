static char help[] = "Test MatMatMult(), MatTranspose(), MatTransposeMatMult() for Dense and Elemental matrices.\n\n";
/*
 Example:
   mpiexec -n <np> ./ex104 -mat_type elemental
*/

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            A,B,C,D;
  PetscInt       i,M=10,N=5,j,nrows,ncols;
  PetscErrorCode ierr;
  PetscRandom    r;
  PetscBool      equal,iselemental;
  PetscReal      fill = 1.0;
  IS             isrows,iscols;
  const PetscInt *rows,*cols;
  PetscScalar    *v,rval;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = PetscOptionsGetInt(NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,PETSC_DECIDE,PETSC_DECIDE,M,N);CHKERRQ(ierr);
  ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_WORLD,&r);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(r);CHKERRQ(ierr);

  /* Set local matrix entries */
  ierr = MatGetOwnershipIS(A,&isrows,&iscols);CHKERRQ(ierr);
  ierr = ISGetLocalSize(isrows,&nrows);CHKERRQ(ierr);
  ierr = ISGetIndices(isrows,&rows);CHKERRQ(ierr);
  ierr = ISGetLocalSize(iscols,&ncols);CHKERRQ(ierr);
  ierr = ISGetIndices(iscols,&cols);CHKERRQ(ierr);
  ierr = PetscMalloc1(nrows*ncols,&v);CHKERRQ(ierr);
  for (i=0; i<nrows; i++) {
    for (j=0; j<ncols; j++) {
      ierr         = PetscRandomGetValue(r,&rval);CHKERRQ(ierr);
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
  ierr = PetscRandomDestroy(&r);CHKERRQ(ierr);

  /* Test MatMatMult() */
  ierr = MatTranspose(A,MAT_INITIAL_MATRIX,&B);CHKERRQ(ierr); /* B = A^T */
  ierr = MatMatMult(B,A,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr); /* C = B*A = A^T*A */

  ierr = MatMatMultSymbolic(B,A,fill,&D);CHKERRQ(ierr); /* D = B*A = A^T*A */
  for (i=0; i<2; i++) {
    /* Repeat the numeric product to test reuse of the previous symbolic product */
    ierr = MatMatMultNumeric(B,A,D);CHKERRQ(ierr);
  }
  ierr = MatMultEqual(C,D,10,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"C != D");
  ierr = MatDestroy(&D);CHKERRQ(ierr);

  /* Test MatTransposeMatMult() */
  ierr = PetscObjectTypeCompare((PetscObject)A,MATELEMENTAL,&iselemental);CHKERRQ(ierr);
  if (!iselemental) {
    ierr = MatTransposeMatMult(A,A,MAT_INITIAL_MATRIX,fill,&D);CHKERRQ(ierr); /* D = A^T*A */
    ierr = MatMultEqual(C,D,10,&equal);CHKERRQ(ierr);
    if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"C != D");
    ierr = MatDestroy(&D);CHKERRQ(ierr);
  }

  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&B);
  ierr = MatDestroy(&A);
  PetscFinalize();
  return(0);
}
