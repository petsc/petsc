/*$Id: ex66.c,v 1.9 2000/10/24 20:26:04 bsmith Exp bsmith $*/

static char help[] = 
"Reads in rectangular matrix from disk, stored from ex65.c\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int        ierr;
  Mat        A;
  PetscViewer     fd;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read matrix and RHS */
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"rect",PETSC_BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatLoad(fd,MATSEQAIJ,&A);CHKERRA(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRA(ierr);

  /* Free data structures */
  ierr = MatDestroy(A);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

