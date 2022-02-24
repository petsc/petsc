static char help[] = "Demonstrates HDF5 vector input/ouput\n\n";

/*T
   Concepts: viewers
   Concepts: HDF5
   Processors: n
T*/
#include <petscsys.h>
#include <petscdm.h>
#include <petscdmda.h>
#include <petscviewerhdf5.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;
  DM             da;
  Vec            global,local,global2;
  PetscMPIInt    rank;
  PetscBool      flg;
  PetscInt       ndof;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help,
                 it prints the various options that can be applied at
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.
  */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* Get number of DOF's from command line */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"DMDA VecView/VecLoad example","");CHKERRQ(ierr);
  {
    ndof = 1;
    PetscOptionsBoundedInt("-ndof","Number of DOF's in DMDA","",ndof,&ndof,NULL,1);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  /* Create a DMDA and an associated vector */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,100,90,PETSC_DECIDE,PETSC_DECIDE,ndof,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMCreateLocalVector(da,&local));
  CHKERRQ(VecSet(global,-1.0));
  CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRQ(VecScale(local,rank+1));
  CHKERRQ(DMLocalToGlobalBegin(da,local,ADD_VALUES,global));
  CHKERRQ(DMLocalToGlobalEnd(da,local,ADD_VALUES,global));

  /* Create the HDF5 viewer for writing */
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,"hdf5output.h5",FILE_MODE_WRITE,&viewer));
  CHKERRQ(PetscViewerSetFromOptions(viewer));

  /* Write the Vec without one extra dimension for BS */
  CHKERRQ(PetscViewerHDF5SetBaseDimension2(viewer, PETSC_FALSE));
  CHKERRQ(PetscObjectSetName((PetscObject) global, "noBsDim"));
  CHKERRQ(VecView(global,viewer));

  /* Write the Vec with one extra, 1-sized, dimension for BS */
  CHKERRQ(PetscViewerHDF5SetBaseDimension2(viewer, PETSC_TRUE));
  CHKERRQ(PetscObjectSetName((PetscObject) global, "bsDim"));
  CHKERRQ(VecView(global,viewer));

  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRMPI(MPI_Barrier(PETSC_COMM_WORLD));
  CHKERRQ(VecDuplicate(global,&global2));

  /* Create the HDF5 viewer for reading */
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,"hdf5output.h5",FILE_MODE_READ,&viewer));
  CHKERRQ(PetscViewerSetFromOptions(viewer));

  /* Load the Vec without the BS dim and compare */
  CHKERRQ(PetscObjectSetName((PetscObject) global2, "noBsDim"));
  CHKERRQ(VecLoad(global2,viewer));

  CHKERRQ(VecEqual(global,global2,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error: Vectors are not equal\n"));
  }

  /* Load the Vec with one extra, 1-sized, BS dim and compare */
  CHKERRQ(PetscObjectSetName((PetscObject) global2, "bsDim"));
  CHKERRQ(VecLoad(global2,viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(VecEqual(global,global2,&flg));
  if (!flg) {
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Error: Vectors are not equal\n"));
  }

  /* clean up and exit */
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(VecDestroy(&global2));
  CHKERRQ(DMDestroy(&da));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

      build:
         requires: hdf5

      test:
         nsize: 4

      test:
         nsize: 4
         suffix: 2
         args: -ndof 2
         output_file: output/ex9_1.out

TEST*/
