static char help[] = "Tests binary MatView() for MPIDENSE matrices \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  Mat            A;
  PetscViewer    viewer;
  char           inputfile[256],outputfile[256];
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-inputfile",inputfile,sizeof(inputfile),&flg));
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-outputfile",outputfile,sizeof(outputfile),&flg));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,inputfile,FILE_MODE_READ,&viewer));
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&A));
  CHKERRQ(MatSetType(A,MATDENSE));
  CHKERRQ(MatLoad(A,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,outputfile,FILE_MODE_WRITE,&viewer));
  CHKERRQ(MatView(A,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  ierr = PetscFinalize();
  return ierr;
}
