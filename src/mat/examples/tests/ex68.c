#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex68.c,v 1.3 1999/02/03 04:30:33 bsmith Exp bsmith $";
#endif

static char help[] = "Tests MatReorderForNonzeroDiagonal().\n\n";

#include "mat.h"

int main(int argc,char **argv)
{
  Mat     mat,B;
  int     ierr,i,j;
  Scalar  v;
  IS      isrow,iscol;

  PetscInitialize(&argc,&argv,(char*)0,help);


  /* ------- Assemble matrix, --------- */

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,4,4,0,0,&mat); CHKERRA(ierr);

  /* set anti-diagonal of matrix */
  v = 1.0;
  i = 0; j = 3;
  ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);
  v = 2.0;
  i = 1; j = 2;
  ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);
  v = 3.0;
  i = 2; j = 1;
  ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);
  v = 4.0;
  i = 3; j = 0;
  ierr = MatSetValues(mat,1,&i,1,&j,&v,INSERT_VALUES); CHKERRA(ierr);

  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  printf("Original matrix\n");
  ierr = ViewerSetFormat(VIEWER_STDOUT_SELF,VIEWER_FORMAT_ASCII_DENSE,0);CHKERRA(ierr);
  ierr = MatView(mat,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  ierr = MatGetOrdering(mat,ORDER_NATURAL,&isrow,&iscol);CHKERRA(ierr);

  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRA(ierr);
  printf("Original matrix permuted by identity\n"); 
  ierr = MatView(B,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = MatDestroy(B);CHKERRA(ierr);

  ierr = MatReorderForNonzeroDiagonal(mat,1.e-8,isrow,iscol);CHKERRA(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRA(ierr);
  printf("Original matrix permuted by identity + NonzeroDiagonal()\n"); 
  ierr = MatView(B,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  printf("Row permutation\n"); 
  ierr = ISView(isrow,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  printf("Column permutation\n"); 
  ierr = ISView(iscol,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = MatDestroy(B);CHKERRA(ierr);

  ierr = ISDestroy(isrow); CHKERRA(ierr);
  ierr = ISDestroy(iscol); CHKERRA(ierr);

  ierr = MatGetOrdering(mat,ORDER_ND,&isrow,&iscol);CHKERRA(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRA(ierr);
  printf("Original matrix permuted by ND\n"); 
  ierr = MatView(B,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = MatDestroy(B);CHKERRA(ierr);
  printf("ND row permutation\n"); 
  ierr = ISView(isrow,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  printf("ND column permutation\n"); 
  ierr = ISView(iscol,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  ierr = MatReorderForNonzeroDiagonal(mat,1.e-8,isrow,iscol);CHKERRA(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRA(ierr);
  printf("Original matrix permuted by ND + NonzeroDiagonal()\n"); 
  ierr = MatView(B,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = MatDestroy(B);CHKERRA(ierr);
  printf("ND + NonzeroDiagonal() row permutation\n"); 
  ierr = ISView(isrow,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  printf("ND + NonzeroDiagonal() column permutation\n"); 
  ierr = ISView(iscol,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  ierr = ISDestroy(isrow); CHKERRA(ierr);
  ierr = ISDestroy(iscol); CHKERRA(ierr);

  ierr = MatGetOrdering(mat,ORDER_RCM,&isrow,&iscol);CHKERRA(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRA(ierr);
  printf("Original matrix permuted by RCM\n"); 
  ierr = MatView(B,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = MatDestroy(B);CHKERRA(ierr);
  printf("RCM row permutation\n"); 
  ierr = ISView(isrow,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  printf("RCM column permutation\n"); 
  ierr = ISView(iscol,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  ierr = MatReorderForNonzeroDiagonal(mat,1.e-8,isrow,iscol);CHKERRA(ierr);
  ierr = MatPermute(mat,isrow,iscol,&B);CHKERRA(ierr);
  printf("Original matrix permuted by RCM + NonzeroDiagonal()\n"); 
  ierr = MatView(B,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = MatDestroy(B);CHKERRA(ierr);
  printf("RCM + NonzeroDiagonal() row permutation\n"); 
  ierr = ISView(isrow,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  printf("RCM + NonzeroDiagonal() column permutation\n"); 
  ierr = ISView(iscol,VIEWER_STDOUT_SELF);CHKERRA(ierr);

   ierr = MatLUFactor(mat,isrow,iscol,0.0);CHKERRA(ierr); 
  printf("Factored matrix permuted by RCM + NonzeroDiagonal()\n"); 
  ierr = MatView(mat,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  /* Free data structures */  
  ierr = ISDestroy(isrow); CHKERRA(ierr);
  ierr = ISDestroy(iscol); CHKERRA(ierr);
  ierr = MatDestroy(mat); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
