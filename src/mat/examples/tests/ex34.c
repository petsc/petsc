/*$Id: ex34.c,v 1.17 2001/04/10 19:35:44 bsmith Exp $*/

static char help[] = "Reads a matrix and vector from a file and writes to another. Input options:\n\
  -fin <input_file> : file to load.  For an example of a 5X5 5-pt. stencil,\n\
                      use the file matbinary.ex.\n\
  -fout <output_file> : file for saving output matrix and vector\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  int         ierr;
  PetscTruth  flg;
  Vec         x;
  Mat         A;
  char        file[256];
  PetscViewer fd;

  PetscInitialize(&argc,&args,(char *)0,help);

  /* Read matrix and RHS */
  ierr = PetscOptionsGetString(PETSC_NULL,"-fin",file,255,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,help);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,PETSC_FILE_RDONLY,&fd);CHKERRQ(ierr);
  ierr = MatLoad(fd,MATSEQAIJ,&A);CHKERRQ(ierr);
  ierr = VecLoad(fd,PETSC_NULL,&x);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);

  /* Write matrix and vector */
  ierr = PetscOptionsGetString(PETSC_NULL,"-fout",file,255,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,help);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,PETSC_FILE_CREATE,&fd);CHKERRQ(ierr);
  ierr = MatView(A,fd);CHKERRQ(ierr);
  ierr = VecView(x,fd);CHKERRQ(ierr);

  /* Free data structures */
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(fd);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}

