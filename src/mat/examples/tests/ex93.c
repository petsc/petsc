static char help[] = "Testing Matrix-Matrix multiplication for SeqAIJ matrices.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  Mat A,B,C,D;
  double a[]={1,1,0,0,1,1,0,0,1};
  int ij[]={0,1,2};
  PetscScalar none=-1.;
  int ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MatCreate(PETSC_COMM_SELF,3,3,3,3,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES);CHKERRQ(ierr);

  ierr = MatSetValues(A,3,ij,3,ij,a,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatTranspose(A,&B);CHKERRQ(ierr);
  ierr = MatMatMult(B,A,&C);CHKERRQ(ierr);
  ierr = MatMatMult(C,A,&D);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);

  ierr = MatDuplicate(A,MAT_COPY_VALUES,&B);CHKERRQ(ierr);
  ierr = MatSeqAIJPtAP(A,B,&C);CHKERRQ(ierr);

  ierr = MatAXPY(&none,C,D,DIFFERENT_NONZERO_PATTERN);CHKERRQ(ierr);
  ierr = MatView(D,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(A);
  ierr = MatDestroy(B);
  ierr = MatDestroy(C);
  ierr = MatDestroy(D);
  PetscFinalize();
  return(0);
}
