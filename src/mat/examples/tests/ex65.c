/*$Id: ex65.c,v 1.9 2001/01/15 21:46:09 bsmith Exp bsmith $*/

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

  ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,m,n,20,0,&A);CHKERRQ(ierr);

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
    ierr = MatSetValues(A,1,&i,cnt,js,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"rect",PETSC_BINARY_CREATE,&view);CHKERRQ(ierr);
  ierr = MatView(A,view);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(view);CHKERRQ(ierr);

  ierr = MatDestroy(A);CHKERRQ(ierr);

  PetscFinalize();
  return 0;
}

