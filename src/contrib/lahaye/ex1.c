static char help[] = "Testing Matrix-Matrix multiplication for SeqAIJ matrices.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  Mat A,B,C;
  double a[]={1,1,0,0,1,1,0,0,1};
  int ij[]={0,1,2};
  int ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,3,3,PETSC_DEFAULT,PETSC_NULL,&A);CHKERRQ(ierr);
  ierr = MatSetOption(A,MAT_IGNORE_ZERO_ENTRIES);CHKERRQ(ierr);

  ierr = MatSetValues(A,3,ij,3,ij,a,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatTranspose(A,&B);CHKERRQ(ierr);
  ierr = MatMatMult_SeqAIJ_SeqAIJ(A,B,&C);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(A);
  ierr = MatDestroy(B);
  ierr = MatDestroy(C);

  PetscFinalize();
  return(0);
}
