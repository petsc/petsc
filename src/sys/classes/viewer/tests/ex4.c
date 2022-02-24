
static char help[] = "Tests PetscOptionsGetViewer() via checking output of PetscViewerASCIIPrintf().\n\n";

#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscViewer       viewer;
  PetscErrorCode    ierr;
  PetscViewerFormat format;
  PetscBool         iascii;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-myviewer",&viewer,&format,NULL));
  CHKERRQ(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Testing PetscViewerASCIIPrintf %d\n", 0));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-myviewer",&viewer,&format,NULL));
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Testing PetscViewerASCIIPrintf %d\n", 1));
    CHKERRQ(PetscViewerPopFormat(viewer));
  }
  CHKERRQ(PetscViewerDestroy(&viewer));
  ierr = PetscFinalize();
  return ierr;
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
