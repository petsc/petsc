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
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  /* Get number of DOF's from command line */
  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,NULL,"DMDA VecView/VecLoad example","");PetscCall(ierr);
  {
    ndof = 1;
    PetscOptionsBoundedInt("-ndof","Number of DOF's in DMDA","",ndof,&ndof,NULL,1);
  }
  ierr = PetscOptionsEnd();PetscCall(ierr);

  /* Create a DMDA and an associated vector */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,100,90,PETSC_DECIDE,PETSC_DECIDE,ndof,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da,&global));
  PetscCall(DMCreateLocalVector(da,&local));
  PetscCall(VecSet(global,-1.0));
  PetscCall(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCall(VecScale(local,rank+1));
  PetscCall(DMLocalToGlobalBegin(da,local,ADD_VALUES,global));
  PetscCall(DMLocalToGlobalEnd(da,local,ADD_VALUES,global));

  /* Create the HDF5 viewer for writing */
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,"hdf5output.h5",FILE_MODE_WRITE,&viewer));
  PetscCall(PetscViewerSetFromOptions(viewer));

  /* Write the Vec without one extra dimension for BS */
  PetscCall(PetscViewerHDF5SetBaseDimension2(viewer, PETSC_FALSE));
  PetscCall(PetscObjectSetName((PetscObject) global, "noBsDim"));
  PetscCall(VecView(global,viewer));

  /* Write the Vec with one extra, 1-sized, dimension for BS */
  PetscCall(PetscViewerHDF5SetBaseDimension2(viewer, PETSC_TRUE));
  PetscCall(PetscObjectSetName((PetscObject) global, "bsDim"));
  PetscCall(VecView(global,viewer));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCallMPI(MPI_Barrier(PETSC_COMM_WORLD));
  PetscCall(VecDuplicate(global,&global2));

  /* Create the HDF5 viewer for reading */
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,"hdf5output.h5",FILE_MODE_READ,&viewer));
  PetscCall(PetscViewerSetFromOptions(viewer));

  /* Load the Vec without the BS dim and compare */
  PetscCall(PetscObjectSetName((PetscObject) global2, "noBsDim"));
  PetscCall(VecLoad(global2,viewer));

  PetscCall(VecEqual(global,global2,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error: Vectors are not equal\n"));
  }

  /* Load the Vec with one extra, 1-sized, BS dim and compare */
  PetscCall(PetscObjectSetName((PetscObject) global2, "bsDim"));
  PetscCall(VecLoad(global2,viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(VecEqual(global,global2,&flg));
  if (!flg) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Error: Vectors are not equal\n"));
  }

  /* clean up and exit */
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(VecDestroy(&global2));
  PetscCall(DMDestroy(&da));

  PetscCall(PetscFinalize());
  return 0;
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
