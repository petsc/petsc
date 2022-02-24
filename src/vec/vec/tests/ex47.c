
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
  char           filename[PETSC_MAX_PATH_LEN];
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetString(NULL,NULL,"-filename",filename,sizeof(filename),&flg));
  if (!flg) CHKERRQ(PetscStrcpy(filename,"x.h5"));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecSetSizes(x,11,PETSC_DETERMINE));
  CHKERRQ(VecSet(x,22.3));

  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&H5viewer));
  CHKERRQ(PetscViewerSetFromOptions(H5viewer));

  /* Write the Vec without one extra dimension for BS */
  CHKERRQ(PetscViewerHDF5SetBaseDimension2(H5viewer, PETSC_FALSE));
  CHKERRQ(PetscObjectSetName((PetscObject) x, "noBsDim"));
  CHKERRQ(VecView(x,H5viewer));

  /* Write the Vec with one extra, 1-sized, dimension for BS */
  CHKERRQ(PetscViewerHDF5SetBaseDimension2(H5viewer, PETSC_TRUE));
  CHKERRQ(PetscObjectSetName((PetscObject) x, "bsDim"));
  CHKERRQ(VecView(x,H5viewer));

  CHKERRQ(PetscViewerDestroy(&H5viewer));
  CHKERRQ(VecDuplicate(x,&y));

  /* Create the HDF5 viewer for reading */
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&H5viewer));
  CHKERRQ(PetscViewerSetFromOptions(H5viewer));

  /* Load the Vec without the BS dim and compare */
  CHKERRQ(PetscObjectSetName((PetscObject) y, "noBsDim"));
  CHKERRQ(VecLoad(y,H5viewer));
  CHKERRQ(VecAXPY(y,-1.0,x));
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  CHKERRQ(VecNorm(x,NORM_2,&dnorm));
  PetscCheckFalse(norm/dnorm > 1.e-6,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec read in 'noBsDim' does not match vector written out %g",(double)(norm/dnorm));

  /* Load the Vec with one extra, 1-sized, BS dim and compare */
  CHKERRQ(PetscObjectSetName((PetscObject) y, "bsDim"));
  CHKERRQ(VecLoad(y,H5viewer));
  CHKERRQ(VecAXPY(y,-1.0,x));
  CHKERRQ(VecNorm(y,NORM_2,&norm));
  CHKERRQ(VecNorm(x,NORM_2,&dnorm));
  PetscCheckFalse(norm/dnorm > 1.e-6,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec read in 'bsDim' does not match vector written out %g",(double)(norm/dnorm));

  CHKERRQ(PetscViewerDestroy(&H5viewer));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&x));
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
