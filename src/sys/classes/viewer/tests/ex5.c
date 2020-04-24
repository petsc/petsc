
static char help[] = "Tests PetscOptionsPushGetViewerOff() via checking output of PetscViewerASCIIPrintf().\n\n";

#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscViewer       viewer;
  PetscErrorCode    ierr;
  PetscViewerFormat format;
  PetscBool         iascii;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-myviewer",&viewer,&format,NULL);CHKERRQ(ierr);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    PetscBool flg;
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Testing PetscViewerASCIIPrintf %d\n", 0);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
    ierr = PetscOptionsPushGetViewerOff(PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-myviewer",&viewer,&format,&flg);CHKERRQ(ierr);
    if (flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONGSTATE,"Pushed viewer off, but viewer was set\n");
    if (viewer) {
      ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
      ierr = PetscViewerASCIIPrintf(viewer,"Testing PetscViewerASCIIPrintf %d\n", 1);CHKERRQ(ierr);
      ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
    }
    ierr = PetscOptionsPopGetViewerOff();CHKERRQ(ierr);
    ierr = PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-myviewer",&viewer,&format,&flg);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,format);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"Testing PetscViewerASCIIPrintf %d\n", 2);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      args: -myviewer

TEST*/
