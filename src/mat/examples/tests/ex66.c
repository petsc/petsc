#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex66.c,v 1.3 1998/11/20 15:29:36 bsmith Exp bsmith $";
#endif

static char help[] = 
"Reads in rectangular matrix from disk, stored from ex65.c\n\n";

#include "mat.h"

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
  ierr = MatLoad(fd,MATSEQAIJ,&A); CHKERRA(ierr);
  ierr = ViewerDestroy(fd); CHKERRA(ierr);

  /* Free data structures */
  ierr = MatDestroy(A); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

