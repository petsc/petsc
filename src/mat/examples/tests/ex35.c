#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex35.c,v 1.6 1999/03/19 21:19:59 bsmith Exp balay $";
#endif

static char help[] = "Tests MatGetSubMatrices().\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat    A,B,*Bsub;
  int    i, j, m = 6, n = 6, N = 36,ierr, I, J;
  Scalar v;
  IS     isrow;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,N,N,5,PETSC_NULL,&A);CHKERRA(ierr);
  for ( i=0; i<m; i++ ) {
    for ( j=0; j<n; j++ ) {
      v = -1.0;  I = j + n*i;
      if ( i>0 )   {J = I - n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( i<m-1 ) {J = I + n; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j>0 )   {J = I - 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      if ( j<n-1 ) {J = I + 1; MatSetValues(A,1,&I,1,&J,&v,INSERT_VALUES);}
      v = 4.0; ierr = MatSetValues(A,1,&I,1,&I,&v,INSERT_VALUES);CHKERRA(ierr);
    }
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatView(A,VIEWER_STDOUT_SELF);CHKERRA(ierr);

  /* take the first diagonal block */
  ierr = ISCreateStride(PETSC_COMM_WORLD,m,0,1,&isrow);CHKERRA(ierr);
  ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub);CHKERRA(ierr);
  B = *Bsub; PetscFree(Bsub);
  ierr = ISDestroy(isrow);CHKERRA(ierr);
  ierr = MatView(B,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = MatDestroy(B);CHKERRA(ierr);

  /* take a strided block */
  ierr = ISCreateStride(PETSC_COMM_WORLD,m,0,2,&isrow);CHKERRA(ierr);
  ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub);CHKERRA(ierr);
  B = *Bsub; PetscFree(Bsub);
  ierr = ISDestroy(isrow);CHKERRA(ierr);
  ierr = MatView(B,VIEWER_STDOUT_SELF);CHKERRA(ierr);
  ierr = MatDestroy(B);CHKERRA(ierr);

  /* take the last block */
  ierr = ISCreateStride(PETSC_COMM_WORLD,m,N-m-1,1,&isrow);CHKERRA(ierr);
  ierr = MatGetSubMatrices(A,1,&isrow,&isrow,MAT_INITIAL_MATRIX,&Bsub);CHKERRA(ierr);
  B = *Bsub; PetscFree(Bsub);
  ierr = ISDestroy(isrow);CHKERRA(ierr);
  ierr = MatView(B,VIEWER_STDOUT_SELF);CHKERRA(ierr);
 
  ierr = MatDestroy(B);CHKERRA(ierr);
  ierr = MatDestroy(A);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

