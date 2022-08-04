static char help[] = "Tests binary MatView() for MPIDENSE matrices \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A;
  PetscViewer    viewer;
  char           inputfile[256],outputfile[256];
  PetscBool      flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-inputfile",inputfile,sizeof(inputfile),&flg));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-outputfile",outputfile,sizeof(outputfile),&flg));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,inputfile,FILE_MODE_READ,&viewer));
  PetscCall(MatCreate(PETSC_COMM_WORLD,&A));
  PetscCall(MatSetType(A,MATDENSE));
  PetscCall(MatLoad(A,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD,outputfile,FILE_MODE_WRITE,&viewer));
  PetscCall(MatView(A,viewer));
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFinalize());
  return 0;
}
