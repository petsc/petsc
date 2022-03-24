/*
   Demonstrates using the HDF5 viewer with a DMDA Vec
 - create a global vector containing a gauss profile (exp(-x^2-y^2))
 - write the global vector in a hdf5 file

   The resulting file gauss.h5 can be viewed with Visit (an open source visualization package)
   Or with some versions of MATLAB with data=hdfread('gauss.h5','pressure'); mesh(data);

   The file storage of the vector is independent of the number of processes used.
 */

#include <petscdm.h>
#include <petscdmda.h>
#include <petscsys.h>
#include <petscviewerhdf5.h>

static char help[] = "Test to write HDF5 file from PETSc DMDA Vec.\n\n";

int main(int argc,char **argv)
{
  DM             da2D;
  PetscInt       i,j,ixs, ixm, iys, iym;
  PetscViewer    H5viewer;
  PetscScalar    xm    = -1.0, xp=1.0;
  PetscScalar    ym    = -1.0, yp=1.0;
  PetscScalar    value = 1.0,dx,dy;
  PetscInt       Nx    = 40, Ny=40;
  Vec            gauss,input;
  PetscScalar    **gauss_ptr;
  PetscReal      norm;
  const char     *vecname;

  dx=(xp-xm)/(Nx-1);
  dy=(yp-ym)/(Ny-1);

  /* Initialize the Petsc context */
  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_STAR,Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da2D));
  CHKERRQ(DMSetFromOptions(da2D));
  CHKERRQ(DMSetUp(da2D));

  /* Set the coordinates */
  CHKERRQ(DMDASetUniformCoordinates(da2D, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0));

  /* Declare gauss as a DMDA component */
  CHKERRQ(DMCreateGlobalVector(da2D,&gauss));
  CHKERRQ(PetscObjectSetName((PetscObject) gauss, "pressure"));

  /* Initialize vector gauss with a constant value (=1) */
  CHKERRQ(VecSet(gauss,value));

  /* Get the coordinates of the corners for each process */
  CHKERRQ(DMDAGetCorners(da2D, &ixs, &iys, 0, &ixm, &iym, 0));

  /* Build the gaussian profile (exp(-x^2-y^2)) */
  CHKERRQ(DMDAVecGetArray(da2D,gauss,&gauss_ptr));
  for (j=iys; j<iys+iym; j++) {
    for (i=ixs; i<ixs+ixm; i++) {
      gauss_ptr[j][i]=PetscExpScalar(-(xm+i*dx)*(xm+i*dx)-(ym+j*dy)*(ym+j*dy));
    }
  }
  CHKERRQ(DMDAVecRestoreArray(da2D,gauss,&gauss_ptr));

  /* Create the HDF5 viewer */
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,"gauss.h5",FILE_MODE_WRITE,&H5viewer));
  CHKERRQ(PetscViewerSetFromOptions(H5viewer));

  /* Write the H5 file */
  CHKERRQ(VecView(gauss,H5viewer));

  /* Close the viewer */
  CHKERRQ(PetscViewerDestroy(&H5viewer));

  CHKERRQ(VecDuplicate(gauss,&input));
  CHKERRQ(PetscObjectGetName((PetscObject)gauss,&vecname));
  CHKERRQ(PetscObjectSetName((PetscObject)input,vecname));

  /* Create the HDF5 viewer for reading */
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD,"gauss.h5",FILE_MODE_READ,&H5viewer));
  CHKERRQ(PetscViewerSetFromOptions(H5viewer));
  CHKERRQ(VecLoad(input,H5viewer));
  CHKERRQ(PetscViewerDestroy(&H5viewer));

  CHKERRQ(VecAXPY(input,-1.0,gauss));
  CHKERRQ(VecNorm(input,NORM_2,&norm));
  PetscCheckFalse(norm > 1.e-6,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"Vec read in does not match vector written out");

  CHKERRQ(VecDestroy(&input));
  CHKERRQ(VecDestroy(&gauss));
  CHKERRQ(DMDestroy(&da2D));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

      build:
         requires: hdf5 !defined(PETSC_USE_CXXCOMPLEX)

      test:
         nsize: 4

      test:
         nsize: 4
         suffix: 2
         args: -viewer_hdf5_base_dimension2
         output_file: output/ex10_1.out

      test:
         nsize: 4
         suffix: 3
         args: -viewer_hdf5_sp_output
         output_file: output/ex10_1.out

TEST*/
