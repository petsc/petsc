static char help[] = "Tests binary MatView() for MPIDENSE matrices \n\n";

#include <petscmat.h>


int main(int argc,char **args)
{
  PetscErrorCode ierr;
  Mat            A;
  PetscViewer    viewer;
  char           inputfile[256],outputfile[256];
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-inputfile",inputfile,sizeof(inputfile),&flg);CHKERRQ(ierr);
  ierr = PetscOptionsGetString(NULL,NULL,"-outputfile",outputfile,sizeof(outputfile),&flg);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,inputfile,FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = MatCreate(PETSC_COMM_WORLD,&A);CHKERRQ(ierr);
  ierr = MatSetType(A,MATDENSE);CHKERRQ(ierr);
  ierr = MatLoad(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,outputfile,FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = MatView(A,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}
