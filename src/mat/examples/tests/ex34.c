/*$Id: ex34.c,v 1.12 2000/10/24 20:26:04 bsmith Exp bsmith $*/

static char help[] = 
"Reads a matrix and vector from a file and writes to another. Input options:\n\
  -fin <input_file> : file to load.  For an example of a 5X5 5-pt. stencil,\n\
                      use the file matbinary.ex.\n\
  -fout <output_file> : file for saving output matrix and vector\n\n";

#include "petscmat.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **args)
{
  int        ierr;
  PetscTruth flg;
  Vec        x;
  Mat        A;
  char       file[256];
  PetscViewer     fd;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read matrix and RHS */
  ierr = PetscOptionsGetString(PETSC_NULL,"-fin",file,255,&flg);CHKERRA(ierr);
  if (!flg) SETERRA(1,help);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,PETSC_BINARY_RDONLY,&fd);CHKERRA(ierr);
  ierr = MatLoad(fd,MATSEQAIJ,&A);CHKERRA(ierr);
  ierr = VecLoad(fd,&x);CHKERRA(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRA(ierr);

  /* Write matrix and vector */
  ierr = PetscOptionsGetString(PETSC_NULL,"-fout",file,255,&flg);CHKERRA(ierr);
  if (!flg) SETERRA(1,help);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,PETSC_BINARY_CREATE,&fd);CHKERRA(ierr);
  ierr = MatView(A,fd);CHKERRA(ierr);
  ierr = VecView(x,fd);CHKERRA(ierr);

  /* Free data structures */
  ierr = MatDestroy(A);CHKERRA(ierr);
  ierr = VecDestroy(x);CHKERRA(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}

