/*$Id: ex65.c,v 1.8 2000/05/05 22:16:17 balay Exp bsmith $*/

static char help[] = "Saves a rectangular sparse matrix to disk\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  Mat    A;
  int    m = 100,n = 11,ierr,js[11],i,j,cnt;
  Scalar values[11];
  PetscViewer view;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,20,0,&A);CHKERRA(ierr);

  for (i=0; i<n; i++) values[i] = (double)i;

  for (i=0; i<m; i++) {
    cnt = 0;
    if (i % 2) {
      for (j=0; j<n; j += 2) {
        js[cnt++] = j;
      }
    } else {
      ;
    }
    ierr = MatSetValues(A,1,&i,cnt,js,values,INSERT_VALUES);CHKERRA(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRA(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"rect",PETSC_BINARY_CREATE,&view);CHKERRA(ierr);
  ierr = MatView(A,view);CHKERRA(ierr);
  ierr = PetscViewerDestroy(view);CHKERRA(ierr);

  ierr = MatDestroy(A);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

