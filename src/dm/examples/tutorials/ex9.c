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
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,100,90,PETSC_DECIDE,PETSC_DECIDE,ndof,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);
  ierr = VecSet(global,-1.0);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = VecScale(local,rank+1);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,local,ADD_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,local,ADD_VALUES,global);CHKERRQ(ierr);

  /* Create the HDF5 viewer for writing */
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"hdf5output.h5",FILE_MODE_WRITE,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);

  /* Write the Vec without one extra dimension for BS */
  ierr = PetscViewerHDF5SetBaseDimension2(viewer, PETSC_FALSE);
  ierr = PetscObjectSetName((PetscObject) global, "noBsDim");CHKERRQ(ierr);
  ierr = VecView(global,viewer);CHKERRQ(ierr);

  /* Write the Vec with one extra, 1-sized, dimension for BS */
  ierr = PetscViewerHDF5SetBaseDimension2(viewer, PETSC_TRUE);
  ierr = PetscObjectSetName((PetscObject) global, "bsDim");CHKERRQ(ierr);
  ierr = VecView(global,viewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = MPI_Barrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = VecDuplicate(global,&global2);CHKERRQ(ierr);

  /* Create the HDF5 viewer for reading */
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"hdf5output.h5",FILE_MODE_READ,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(viewer);CHKERRQ(ierr);

  /* Load the Vec without the BS dim and compare */
  ierr = PetscObjectSetName((PetscObject) global2, "noBsDim");CHKERRQ(ierr);
  ierr = VecLoad(global2,viewer);CHKERRQ(ierr);

  ierr = VecEqual(global,global2,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: Vectors are not equal\n");CHKERRQ(ierr);
  }

  /* Load the Vec with one extra, 1-sized, BS dim and compare */
  ierr = PetscObjectSetName((PetscObject) global2, "bsDim");CHKERRQ(ierr);
  ierr = VecLoad(global2,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = VecEqual(global,global2,&flg);CHKERRQ(ierr);
  if (!flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Error: Vectors are not equal\n");CHKERRQ(ierr);
  }

  /* clean up and exit */
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = VecDestroy(&global2);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);

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
