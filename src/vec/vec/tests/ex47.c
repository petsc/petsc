
static char help[] = "Tests PetscViewerHDF5 VecView()/VecLoad() function.\n\n";

#include <petscviewer.h>
#include <petscviewerhdf5.h>
#include <petscvec.h>

int main(int argc,char **args)
{
  Vec            x,y;
  PetscReal      norm,dnorm;
  PetscViewer    H5viewer;
  char           filename[PETSC_MAX_PATH_LEN];
  PetscBool      flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PetscOptionsGetString(NULL,NULL,"-filename",filename,sizeof(filename),&flg));
  if (!flg) PetscCall(PetscStrcpy(filename,"x.h5"));
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSetSizes(x,11,PETSC_DETERMINE));
  PetscCall(VecSet(x,22.3));

  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&H5viewer));
  PetscCall(PetscViewerSetFromOptions(H5viewer));

  /* Write the Vec without one extra dimension for BS */
  PetscCall(PetscViewerHDF5SetBaseDimension2(H5viewer, PETSC_FALSE));
  PetscCall(PetscObjectSetName((PetscObject) x, "noBsDim"));
  PetscCall(VecView(x,H5viewer));

  /* Write the Vec with one extra, 1-sized, dimension for BS */
  PetscCall(PetscViewerHDF5SetBaseDimension2(H5viewer, PETSC_TRUE));
  PetscCall(PetscObjectSetName((PetscObject) x, "bsDim"));
  PetscCall(VecView(x,H5viewer));

  PetscCall(PetscViewerDestroy(&H5viewer));
  PetscCall(VecDuplicate(x,&y));

  /* Create the HDF5 viewer for reading */
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD,filename,FILE_MODE_READ,&H5viewer));
  PetscCall(PetscViewerSetFromOptions(H5viewer));

  /* Load the Vec without the BS dim and compare */
  PetscCall(PetscObjectSetName((PetscObject) y, "noBsDim"));
  PetscCall(VecLoad(y,H5viewer));
  PetscCall(VecAXPY(y,-1.0,x));
  PetscCall(VecNorm(y,NORM_2,&norm));
  PetscCall(VecNorm(x,NORM_2,&dnorm));
  PetscCheck(norm/dnorm <= 1.e-6,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec read in 'noBsDim' does not match vector written out %g",(double)(norm/dnorm));

  /* Load the Vec with one extra, 1-sized, BS dim and compare */
  PetscCall(PetscObjectSetName((PetscObject) y, "bsDim"));
  PetscCall(VecLoad(y,H5viewer));
  PetscCall(VecAXPY(y,-1.0,x));
  PetscCall(VecNorm(y,NORM_2,&norm));
  PetscCall(VecNorm(x,NORM_2,&dnorm));
  PetscCheck(norm/dnorm <= 1.e-6,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec read in 'bsDim' does not match vector written out %g",(double)(norm/dnorm));

  PetscCall(PetscViewerDestroy(&H5viewer));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
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
