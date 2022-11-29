
static char help[] = "Tests PetscOptionsPushGetViewerOff() via checking output of PetscViewerASCIIPrintf().\n\n";

#include <petscviewer.h>

int main(int argc, char **args)
{
  PetscViewer       viewer;
  PetscViewerFormat format;
  PetscBool         iascii;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetViewer(PETSC_COMM_WORLD, NULL, NULL, "-myviewer", &viewer, &format, NULL));
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscBool flg;
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Testing PetscViewerASCIIPrintf %d\n", 0));
    PetscCall(PetscViewerPopFormat(viewer));
    PetscCall(PetscViewerDestroy(&viewer));
    PetscCall(PetscOptionsPushGetViewerOff(PETSC_TRUE));
    PetscCall(PetscOptionsGetViewer(PETSC_COMM_WORLD, NULL, NULL, "-myviewer", &viewer, &format, &flg));
    PetscCheck(!flg, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONGSTATE, "Pushed viewer off, but viewer was set");
    if (viewer) {
      PetscCall(PetscViewerPushFormat(viewer, format));
      PetscCall(PetscViewerASCIIPrintf(viewer, "Testing PetscViewerASCIIPrintf %d\n", 1));
      PetscCall(PetscViewerPopFormat(viewer));
    }
    PetscCall(PetscOptionsPopGetViewerOff());
    PetscCall(PetscOptionsGetViewer(PETSC_COMM_WORLD, NULL, NULL, "-myviewer", &viewer, &format, &flg));
    PetscCall(PetscViewerPushFormat(viewer, format));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Testing PetscViewerASCIIPrintf %d\n", 2));
    PetscCall(PetscViewerPopFormat(viewer));
  }
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -myviewer

TEST*/
