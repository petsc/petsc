/*$Id: ex44.c,v 1.15 2001/04/10 19:35:44 bsmith Exp $*/

static char help[] = "Loads matrix dumped by ex43.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat         C;
  PetscViewer viewer;
  int         ierr;

  PetscInitialize(&argc,&args,0,help);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"matrix.dat",PETSC_FILE_RDONLY,&viewer); 
        CHKERRQ(ierr);
  MatLoad(viewer,MATMPIDENSE,&C);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}


