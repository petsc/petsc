static char help[] = "Testing coarse projection of a matrix for SeqAIJ matrices.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  Mat A,P,C;
  int    ai[]={0,2,3,4,6,7};
  int    aj[]={0,4,1,2,1,3,4};
  double aa[]={1,1,1,1,1,1,1};
  int    pi[]={0,1,2,3,4,5};
  int    pj[]={0,1,2,2,2};
  double pa[]={1,1,1,1,1};
  int    ierr;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,5,5,ai,aj,aa,&A);CHKERRQ(ierr);
  ierr = MatCreateSeqAIJWithArrays(PETSC_COMM_SELF,5,3,pi,pj,pa,&P);CHKERRQ(ierr);
  ierr = MatApplyPtAP_SeqAIJ(A,P,&C);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = MatDestroy(P);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);
  PetscFinalize();
  return(0);
}
