static char help[] = "Testing Matrix-Matrix multiplication for SeqAIJ matrices.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  Mat            A,B,C,D;
  PetscScalar    a[]={1.,1.,0.,0.,1.,1.,0.,0.,1.};
  PetscInt       ij[]={0,1,2};
  PetscScalar    none=-1.;
  PetscErrorCode ierr;
  PetscReal      fill=4;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,3,3,3,3);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES);CHKERRQ(ierr);

  ierr = MatSetValues(A,3,ij,3,ij,a,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* Form A^T*A*A to test PtAP routine. */
  ierr = MatTranspose(A,&B);CHKERRQ(ierr);
  ierr = MatMatMult(B,A,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  ierr = MatMatMultSymbolic(C,A,fill,&D);CHKERRQ(ierr);
  ierr = MatMatMultNumeric(C,A,D);CHKERRQ(ierr);
  ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Repeat the numeric product to test reuse of the previous symbolic product */
  ierr = MatMatMultNumeric(C,A,D);CHKERRQ(ierr);
  ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(B);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);

  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  ierr = MatPtAP(A,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);

  ierr = MatAXPY(D,none,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = MatDestroy(D);CHKERRQ(ierr);

  /* Repeat PtAP to test symbolic/numeric separation for reuse of the symbolic product */
  ierr = MatPtAP(A,B,MAT_INITIAL_MATRIX,fill,&C);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatPtAPSymbolic(A,B,fill,&D);CHKERRQ(ierr);
  ierr = MatPtAPNumeric(A,B,D);CHKERRQ(ierr);
  ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Repeat numeric product to test reuse of the previous symbolic product */
  ierr = MatPtAPNumeric(A,B,D);CHKERRQ(ierr);
  ierr = MatAXPY(D,none,C,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(A);
  ierr = MatDestroy(B);
  ierr = MatDestroy(C);
  ierr = MatDestroy(D);
  PetscFinalize();
  return(0);
}
