#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex65.c,v 1.3 1998/12/03 04:01:49 bsmith Exp bsmith $";
#endif

static char help[] = "Saves a rectangular sparse matrix to disk\n\n";

#include "mat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat    A;
  int    m = 100, n = 11, ierr, js[11],i,j,cnt;
  Scalar values[11];
  Viewer view;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,20,0,&A); CHKERRA(ierr);

  for ( i=0; i<n; i++ ) values[i] = (double) i;

  for ( i=0; i<m; i++ ) {
    cnt = 0;
    if ( i % 2 ) {
      for ( j=0; j<n; j += 2) {
        js[cnt++] = j;
      }
    } else {
      ;
    }
    ierr = MatSetValues(A,1,&i,cnt,js,values,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY); CHKERRA(ierr);

  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"rect",BINARY_CREATE,&view);CHKERRA(ierr);
  ierr = MatView(A,view); CHKERRA(ierr);
  ierr = ViewerDestroy(view); CHKERRA(ierr);

  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

