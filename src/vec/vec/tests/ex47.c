
static char help[] = "Tests PetscViewerHDF5 VecView()/VecLoad() function.\n\n";

#include <petscviewer.h>
#include <petscviewerhdf5.h>
#include <petscvec.h>

int main(int argc,char **args)
{
  PetscErrorCode ierr;
  Vec            x,y;
  PetscReal      norm,dnorm;
  PetscViewer    H5viewer;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,11,PETSC_DETERMINE);CHKERRQ(ierr);
  ierr = VecSet(x,22.3);CHKERRQ(ierr);

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"x.h5",FILE_MODE_WRITE,&H5viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(H5viewer);CHKERRQ(ierr);

  /* Write the Vec without one extra dimension for BS */
  ierr = PetscViewerHDF5SetBaseDimension2(H5viewer, PETSC_FALSE);
  ierr = PetscObjectSetName((PetscObject) x, "noBsDim");CHKERRQ(ierr);
  ierr = VecView(x,H5viewer);CHKERRQ(ierr);

  /* Write the Vec with one extra, 1-sized, dimension for BS */
  ierr = PetscViewerHDF5SetBaseDimension2(H5viewer, PETSC_TRUE);
  ierr = PetscObjectSetName((PetscObject) x, "bsDim");CHKERRQ(ierr);
  ierr = VecView(x,H5viewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&H5viewer);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);

  /* Create the HDF5 viewer for reading */
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"x.h5",FILE_MODE_READ,&H5viewer);CHKERRQ(ierr);
  ierr = PetscViewerSetFromOptions(H5viewer);CHKERRQ(ierr);

  /* Load the Vec without the BS dim and compare */
  ierr = PetscObjectSetName((PetscObject) y, "noBsDim");CHKERRQ(ierr);
  ierr = VecLoad(y,H5viewer);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&dnorm);CHKERRQ(ierr);
  if (norm/dnorm > 1.e-6) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec read in 'noBsDim' does not match vector written out %g",(double)(norm/dnorm));

  /* Load the Vec with one extra, 1-sized, BS dim and compare */
  ierr = PetscObjectSetName((PetscObject) y, "bsDim");CHKERRQ(ierr);
  ierr = VecLoad(y,H5viewer);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&dnorm);CHKERRQ(ierr);
  if (norm/dnorm > 1.e-6) SETERRQ1(PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec read in 'bsDim' does not match vector written out %g",(double)(norm/dnorm));

  ierr = PetscViewerDestroy(&H5viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   build:
     requires: hdf5

   test:
     requires: hdf5

   test:
     suffix: 2
     nsize: 4

   test:
     suffix: 3
     nsize: 4
     args: -viewer_hdf5_base_dimension2

   test:
     suffix: 4
     nsize: 4
     args: -viewer_hdf5_sp_output

TEST*/
