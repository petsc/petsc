/*$Id: ex4.c,v 1.14 2000/05/05 22:16:17 balay Exp bsmith $*/

static char help[] = "Creates a matrix, inserts some values, and tests\n\
MatGetSubMatrices and MatZeroEntries.\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Mat       mat,submat,*submatrices;
  int       m = 10,n = 10,i = 4,tmp,ierr;
  IS        irkeep,ickeep;
  Scalar    value = 1.0;
  Viewer    sviewer;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_COMMON,0);CHKERRA(ierr);
  ierr = ViewerSetFormat(VIEWER_STDOUT_SELF,VIEWER_FORMAT_ASCII_COMMON,0);CHKERRA(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,&mat);CHKERRA(ierr);
  ierr = MatSetFromOptions(mat);CHKERRA(ierr);
  for (i=0; i<m; i++) {
    value = (double)i+1; tmp = i % 5; 
    ierr = MatSetValues(mat,1,&tmp,1,&i,&value,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = ViewerASCIIPrintf(VIEWER_STDOUT_WORLD,"Original matrix\n");CHKERRA(ierr);
  ierr = MatView(mat,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* Form submatrix with rows 2-4 and columns 4-8 */
  ierr = ISCreateStride(PETSC_COMM_SELF,3,2,1,&irkeep);CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,5,4,1,&ickeep);CHKERRA(ierr);
  ierr = MatGetSubMatrices(mat,1,&irkeep,&ickeep,MAT_INITIAL_MATRIX,&submatrices);CHKERRA(ierr);
  submat = *submatrices; 
  ierr = PetscFree(submatrices);CHKERRA(ierr);
  /*
     sviewer will cause the submatrices (one per processor) to be printed in the correct order
  */
  ierr = ViewerASCIIPrintf(VIEWER_STDOUT_WORLD,"Submatrices\n");CHKERRA(ierr);
  ierr = ViewerGetSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRA(ierr);
  ierr = MatView(submat,sviewer);CHKERRA(ierr);
  ierr = ViewerRestoreSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRA(ierr);
  ierr = ViewerFlush(VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* Zero the original matrix */
  ierr = ViewerASCIIPrintf(VIEWER_STDOUT_WORLD,"Original zeroed matrix\n");CHKERRA(ierr);
  ierr = MatZeroEntries(mat);CHKERRA(ierr);
  ierr = MatView(mat,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = ISDestroy(irkeep);CHKERRA(ierr);
  ierr = ISDestroy(ickeep);CHKERRA(ierr);
  ierr = MatDestroy(submat);CHKERRA(ierr);
  ierr = MatDestroy(mat);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
