static char help[] = "Tests binary MatView() for MPIDENSE matrices \n\n";

#include "petscmat.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  PetscErrorCode ierr;
  Mat            A;
  PetscViewer    viewer;
  char           inputfile[256],outputfile[256];
  PetscTruth     flg;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetString(PETSC_NULL,"-inputfile",inputfile,256,&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(PETSC_NULL,"-outputfile",outputfile,256,&flg);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inputfile,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatLoad(viewer,MATDENSE,&A);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outputfile,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFormat(viewer,PETSC_VIEWER_NATIVE);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
