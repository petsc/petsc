
static char help[] = "Tests PetscOptionsPushGetViewerOff() via checking output of PetscViewerASCIIPrintf().\n\n";

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
    PetscBool flg;
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Testing PetscViewerASCIIPrintf %d\n", 0));
    CHKERRQ(PetscViewerPopFormat(viewer));
    CHKERRQ(PetscViewerDestroy(&viewer));
    CHKERRQ(PetscOptionsPushGetViewerOff(PETSC_TRUE));
    CHKERRQ(PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-myviewer",&viewer,&format,&flg));
    PetscCheckFalse(flg,PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Pushed viewer off, but viewer was set");
    if (viewer) {
      CHKERRQ(PetscViewerPushFormat(viewer,format));
      CHKERRQ(PetscViewerASCIIPrintf(viewer,"Testing PetscViewerASCIIPrintf %d\n", 1));
      CHKERRQ(PetscViewerPopFormat(viewer));
    }
    CHKERRQ(PetscOptionsPopGetViewerOff());
    CHKERRQ(PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-myviewer",&viewer,&format,&flg));
    CHKERRQ(PetscViewerPushFormat(viewer,format));
    CHKERRQ(PetscViewerASCIIPrintf(viewer,"Testing PetscViewerASCIIPrintf %d\n", 2));
    CHKERRQ(PetscViewerPopFormat(viewer));
  }
  CHKERRQ(PetscViewerDestroy(&viewer));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -myviewer

TEST*/
