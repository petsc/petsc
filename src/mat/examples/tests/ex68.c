
static char help[] = "Tests MatReorderForNonzeroDiagonal().\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Mat            mat,B;
  PetscErrorCode ierr;
  PetscInt       i,j;
  PetscScalar    v;
  IS             isrow,iscol;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 


  /* ------- Assemble matrix, --------- */

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,4,4,0,0,&mat);CHKERRQ(ierr);

  /* set anti-diagonal of matrix */
  v = 1.0;
  i = 0; j = 3;
  ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  v = 2.0;
  i = 1; j = 2;
  ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  v = 3.0;
  i = 2; j = 1;
  ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  v = 4.0;
  i = 3; j = 0;
  ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  printf("Original matrix\n");
  ierr = PetscViewerSetFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_DENSE);CHKERRQ(ierr);
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = MatGetOrdering(mat,MATORDERING_NATURAL,&isrow,&iscol);CHKERRQ(ierr);

  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRQ(ierr);
  printf("Original matrix permuted by identity\n"); 
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);

  ierr = MatReorderForNonzeroDiagonal(mat,1.e-8,isrow,iscol);CHKERRQ(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRQ(ierr);
  printf("Original matrix permuted by identity + NonzeroDiagonal()\n"); 
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  printf("Row permutation\n"); 
  ierr = ISView(isrow,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  printf("Column permutation\n"); 
  ierr = ISView(iscol,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);

  ierr = ISDestroy(isrow);CHKERRQ(ierr);
  ierr = ISDestroy(iscol);CHKERRQ(ierr);

  ierr = MatGetOrdering(mat,MATORDERING_ND,&isrow,&iscol);CHKERRQ(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRQ(ierr);
  printf("Original matrix permuted by ND\n"); 
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  printf("ND row permutation\n"); 
  ierr = ISView(isrow,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  printf("ND column permutation\n"); 
  ierr = ISView(iscol,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = MatReorderForNonzeroDiagonal(mat,1.e-8,isrow,iscol);CHKERRQ(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRQ(ierr);
  printf("Original matrix permuted by ND + NonzeroDiagonal()\n"); 
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  printf("ND + NonzeroDiagonal() row permutation\n"); 
  ierr = ISView(isrow,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  printf("ND + NonzeroDiagonal() column permutation\n"); 
  ierr = ISView(iscol,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = ISDestroy(isrow);CHKERRQ(ierr);
  ierr = ISDestroy(iscol);CHKERRQ(ierr);

  ierr = MatGetOrdering(mat,MATORDERING_RCM,&isrow,&iscol);CHKERRQ(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRQ(ierr);
  printf("Original matrix permuted by RCM\n"); 
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  printf("RCM row permutation\n"); 
  ierr = ISView(isrow,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  printf("RCM column permutation\n"); 
  ierr = ISView(iscol,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = MatReorderForNonzeroDiagonal(mat,1.e-8,isrow,iscol);CHKERRQ(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRQ(ierr);
  printf("Original matrix permuted by RCM + NonzeroDiagonal()\n"); 
  ierr = MatView(B,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatDestroy(B);CHKERRQ(ierr);
  printf("RCM + NonzeroDiagonal() row permutation\n"); 
  ierr = ISView(isrow,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  printf("RCM + NonzeroDiagonal() column permutation\n"); 
  ierr = ISView(iscol,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

   ierr = MatLUFactor(mat,isrow,iscol,PETSC_NULL);CHKERRQ(ierr); 
  printf("Factored matrix permuted by RCM + NonzeroDiagonal()\n"); 
  ierr = MatView(mat,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  /* Free data structures */  
  ierr = ISDestroy(isrow);CHKERRQ(ierr);
  ierr = ISDestroy(iscol);CHKERRQ(ierr);
  ierr = MatDestroy(mat);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
 
