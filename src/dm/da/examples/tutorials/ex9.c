static char help[] = "Demonstrates HD5 vector input/ouput\n\n";

/*T
   Concepts: viewers; HDF5
   Processors: n
T*/
#include "petscsys.h"
#include "petscda.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscViewer    viewer;
  DA             da;
  Vec            global,local,global2;
  PetscMPIInt    rank;
  PetscTruth     flg;

  /*
    Every PETSc routine should begin with the PetscInitialize() routine.
    argc, argv - These command line arguments are taken to extract the options
                 supplied to PETSc and options supplied to MPI.
    help       - When PETSc executable is invoked with the option -help, 
                 it prints the various options that can be applied at 
                 runtime.  The user can use the "help" variable place
                 additional help messages in this printout.
  */
  ierr = PetscInitialize(&argc,&argv,(char *)0,help);CHKERRQ(ierr);

  /* Create a DA and an associated vector */
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,100,90,PETSC_DECIDE,PETSC_DECIDE,2,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRQ(ierr);
  ierr = VecSet(global,-1.0);CHKERRQ(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);
  ierr = VecScale(local,rank+1);CHKERRQ(ierr);
  ierr = DALocalToGlobal(da,local,ADD_VALUES,global);CHKERRQ(ierr);

  /* 
     Write output file with PetscViewerHDF5 viewer. 

  */
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"hdf5output",FILE_MODE_WRITE,&viewer); CHKERRQ(ierr);
  ierr = VecView(global,viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  
  ierr = VecDuplicate(global,&global2);CHKERRQ(ierr);
  ierr = VecCopy(global,global2);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"hdf5output",FILE_MODE_READ,&viewer); CHKERRQ(ierr);
  ierr = VecLoadIntoVector(viewer,global);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = VecEqual(global,global2,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Vectors are equal\n");CHKERRQ(ierr);
  } else {
    ierr = PetscPrintf(PETSC_COMM_WORLD,"Vectors are not equal\n");CHKERRQ(ierr);
  }

  /* clean up and exit */
  ierr = DADestroy(da);CHKERRQ(ierr);
  ierr = VecDestroy(local);CHKERRQ(ierr);
  ierr = VecDestroy(global);CHKERRQ(ierr);
  ierr = VecDestroy(global2);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;  
}
