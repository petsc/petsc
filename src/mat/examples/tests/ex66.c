/*$Id: ex66.c,v 1.7 1999/10/24 14:02:39 bsmith Exp balay $*/

static char help[] = 
"Reads in rectangular matrix from disk, stored from ex65.c\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int        ierr;
  Mat        A;
  Viewer     fd;
  MatType    type;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read matrix and RHS */
  ierr = ViewerBinaryOpen(PETSC_COMM_WORLD,"rect",BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatGetTypeFromOptions(PETSC_COMM_WORLD,PETSC_NULL,&type,PETSC_NULL);CHKERRA(ierr);
  ierr = MatLoad(fd,MATSEQAIJ,&A);CHKERRA(ierr);
  ierr = ViewerDestroy(fd);CHKERRA(ierr);

  /* Free data structures */
  ierr = MatDestroy(A);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

