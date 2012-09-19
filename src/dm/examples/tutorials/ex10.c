/*
   Demonstrates using the HDF5 viewer with a DMDA Vec
 - create a global vector containing a gauss profile (exp(-x^2-y^2))
 - write the global vector in a hdf5 file

   The resulting file gauss.h5 can be viewed with Visit (an open source visualization package)
   Or with some versions of MATLAB with data=hdfread('gauss.h5','pressure'); mesh(data);

   The file storage of the vector is independent of the number of processes used.
 */

#include <math.h>
#include <petscdmda.h>
#include <petscsys.h>

static char help[] = "Test to write HDF5 file from PETSc DMDA Vec.\n\n";

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             da2D;
  PetscInt       i,j,ixs, ixm, iys, iym;;
  PetscViewer    H5viewer;
  PetscScalar    xm=-1.0, xp=1.0;
  PetscScalar    ym=-1.0, yp=1.0;
  PetscScalar    value=1.0,dx,dy;
  PetscInt       Nx=40, Ny=40;
  Vec            gauss;
  PetscScalar    **gauss_ptr;

  dx=(xp-xm)/(Nx-1);
  dy=(yp-ym)/(Ny-1);

  // Initialize the Petsc context
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr);

  // Build of the DMDA
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DMDA_BOUNDARY_NONE, DMDA_BOUNDARY_NONE,DMDA_STENCIL_STAR,Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&da2D);CHKERRQ(ierr);

  // Set the coordinates
  DMDASetUniformCoordinates(da2D, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);

  // Declare gauss as a DMDA component
  ierr = DMCreateGlobalVector(da2D,&gauss);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) gauss, "pressure");CHKERRQ(ierr);

  // Initialize vector gauss with a constant value (=1)
  ierr = VecSet(gauss,value);CHKERRQ(ierr);

  // Get the coordinates of the corners for each process
  ierr = DMDAGetCorners(da2D, &ixs, &iys, 0, &ixm, &iym, 0);CHKERRQ(ierr);

  /* Build the gaussian profile (exp(-x^2-y^2)) */
  ierr = DMDAVecGetArray(da2D,gauss,&gauss_ptr);CHKERRQ(ierr);
  for (j=iys; j<iys+iym; j++){
    for (i=ixs; i<ixs+ixm; i++){
      gauss_ptr[j][i]=exp(-(xm+i*dx)*(xm+i*dx)-(ym+j*dy)*(ym+j*dy));
    }
  }
  ierr = DMDAVecRestoreArray(da2D,gauss,&gauss_ptr);CHKERRQ(ierr);	

  // Create the HDF5 viewer
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"gauss.h5",FILE_MODE_WRITE,&H5viewer); CHKERRQ(ierr);	

  // Write the H5 file
  ierr = VecView(gauss,H5viewer);CHKERRQ(ierr);	

  // Cleaning stage
  ierr = PetscViewerDestroy(&H5viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&gauss);CHKERRQ(ierr);
  ierr = DMDestroy(&da2D);CHKERRQ(ierr);
  ierr = PetscFinalize();
    return 0;
}
