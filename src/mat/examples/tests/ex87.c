/*$Id: ex31.c,v 1.27 2001/08/07 03:03:07 balay Exp $*/

static char help[] = "Tests if MatGetSubMatrices() and MatIncreaseOverlap() for SBAIJ matrices\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat         BAIJ,SBAIJ;
  PetscViewer viewer;
  char        file[128];
  PetscTruth  flg;
  int         ierr;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetString(PETSC_NULL,"-f",file,127,&flg);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,PETSC_FILE_RDONLY,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(viewer,MATBAIJ,&BAIJ);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,file,PETSC_FILE_RDONLY,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(viewer,MATSBAIJ,&SBAIJ);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);

  /* Free data structures */
  ierr = MatDestroy(BAIJ);CHKERRQ(ierr);
  ierr = MatDestroy(SBAIJ);CHKERRQ(ierr);

  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}


