/*
   Demonstrates using the HDF5 viewer with a DA Vec
 - create a global vector containing a gauss profile (exp(-x^2-y^2))
 - write the global vector in a hdf5 file 

   The resulting file gauss.h5 can be viewed with Visit (an open source visualization package)
   Or with some versions of Matlab with data=hdfread('gauss.h5','pressure'); mesh(data);   

   The file storage of the vector is independent of the number of processes used.
 */

#include <math.h>
#include "petscda.h"
#include "petscsys.h"

static char help[] = "Test to write HDF5 file from PETSc DA Vec.\n\n";

int main(int argc,char **argv) 
{
  PetscErrorCode ierr;
  DA             da2D;
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
  
  // Build of the DA
  ierr = DACreate2d(PETSC_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_STAR,Nx,Ny,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&da2D);CHKERRQ(ierr);
  
  // Set the coordinates
  DASetUniformCoordinates(da2D, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
  
  // Declare gauss as a DA component
  ierr = DACreateGlobalVector(da2D,&gauss);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) gauss, "pressure");CHKERRQ(ierr);
  
  // Initialize vector gauss with a constant value (=1)
  ierr = VecSet(gauss,value);CHKERRQ(ierr);
  
  // Get the coordinates of the corners for each process
  ierr = DAGetCorners(da2D, &ixs, &iys, 0, &ixm, &iym, 0);CHKERRQ(ierr);
  
  /* Build the gaussian profile (exp(-x^2-y^2)) */
  ierr = DAVecGetArray(da2D,gauss,&gauss_ptr);CHKERRQ(ierr);
  for (j=iys; j<iys+iym; j++){
    for (i=ixs; i<ixs+ixm; i++){
      gauss_ptr[j][i]=exp(-(xm+i*dx)*(xm+i*dx)-(ym+j*dy)*(ym+j*dy));
    }
  }
  ierr = DAVecRestoreArray(da2D,gauss,&gauss_ptr);CHKERRQ(ierr);	
  
  // Create the HDF5 viewer
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"gauss.h5",FILE_MODE_WRITE,&H5viewer); CHKERRQ(ierr);	
  
  // Write the H5 file
  ierr = VecView(gauss,H5viewer);CHKERRQ(ierr);	
  
  // Cleaning stage
  ierr = PetscViewerDestroy(H5viewer);CHKERRQ(ierr);
  ierr = VecDestroy(gauss);CHKERRQ(ierr);
  ierr = DADestroy(da2D);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
    return 0;
}
