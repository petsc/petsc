
static char help[] = "Tests PetscOptionsGetViewer() via checking output of PetscViewerASCIIPrintf().\n\n";

#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         iascii;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-myviewer",&viewer,&format,NULL));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Testing PetscViewerASCIIPrintf %d\n", 0));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-myviewer",&viewer,&format,NULL));
    PetscCall(PetscViewerPushFormat(viewer,format));
    PetscCall(PetscViewerASCIIPrintf(viewer,"Testing PetscViewerASCIIPrintf %d\n", 1));
    PetscCall(PetscViewerPopFormat(viewer));
  }
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -myviewer ascii:ex4w1.tmp
      filter: cat ex4w1.tmp
      output_file: output/ex4w.out

   test:
      suffix: 2
      args: -myviewer ascii:ex4w2.tmp::
      filter: cat ex4w2.tmp
      output_file: output/ex4w.out

   test:
      suffix: 3
      args: -myviewer ascii:ex4w3.tmp::write
      filter: cat ex4w3.tmp
      output_file: output/ex4w.out

   test:
      suffix: 4
      args: -myviewer ascii:ex4a1.tmp::append
      filter: cat ex4a1.tmp
      output_file: output/ex4a.out

TEST*/
