#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex4.c,v 1.8 1999/04/16 16:07:27 bsmith Exp balay $";
#endif

static char help[] = "Creates a matrix, inserts some values, and tests\n\
MatGetSubMatrices and MatZeroEntries.\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  Mat       mat, submat,*submatrices;
  int       m = 10, n = 10, i = 4, tmp, ierr;
  IS        irkeep, ickeep;
  Scalar    value = 1.0;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = ViewerSetFormat(VIEWER_STDOUT_WORLD,VIEWER_FORMAT_ASCII_COMMON,0);CHKERRA(ierr);

  ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,&mat);CHKERRA(ierr);
  for (i=0; i<m; i++ ) {
    value = (double) i+1; tmp = i % 5; 
    ierr = MatSetValues(mat,1,&tmp,1,&i,&value,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  printf("initial matrix:\n");
  ierr = MatView(mat,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* Form submatrix with rows 2-4 and columns 4-8 */
  ierr = ISCreateStride(PETSC_COMM_SELF,3,2,1,&irkeep);CHKERRA(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,5,4,1,&ickeep);CHKERRA(ierr);
  ierr = MatGetSubMatrices(mat,1,&irkeep,&ickeep,MAT_INITIAL_MATRIX,&submatrices);CHKERRA(ierr);
  submat = *submatrices; PetscFree(submatrices);
  printf("\nsubmatrix:\n");
  ierr = MatView(submat,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  /* Zero the original matrix */
  printf("\nzeroed matrix:\n");
  ierr = MatZeroEntries(mat);CHKERRA(ierr);
  ierr = MatView(mat,VIEWER_STDOUT_WORLD);CHKERRA(ierr);

  ierr = ISDestroy(irkeep);CHKERRA(ierr);
  ierr = ISDestroy(ickeep);CHKERRA(ierr);
  ierr = MatDestroy(submat);CHKERRA(ierr);
  ierr = MatDestroy(mat);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
