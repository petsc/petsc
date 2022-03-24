
static char help[] = "Demonstrates PetscOptionsGetViewer().\n\n";

#include <petscviewer.h>

int main(int argc,char **args)
{
  PetscViewer       viewer;
  PetscViewerFormat format;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PetscOptionsGetViewer(PETSC_COMM_WORLD,NULL,NULL,"-myviewer",&viewer,&format,NULL));
  CHKERRQ(PetscViewerPushFormat(viewer,format));
  CHKERRQ(PetscViewerView(viewer,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerPopFormat(viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -myviewer ascii

   testset:
      args: -myviewer hdf5:my.hdf5:hdf5_xdmf
      requires: hdf5
      test:
        suffix: 2a
        args: -viewer_hdf5_base_dimension2 false -viewer_hdf5_sp_output true  -viewer_hdf5_collective false
      test:
        suffix: 2b
        args: -viewer_hdf5_base_dimension2 true  -viewer_hdf5_sp_output false -viewer_hdf5_collective true

TEST*/
