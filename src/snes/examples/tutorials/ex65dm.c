static char help[] = "DM.\n ./ex61gen [-random_seed <int>] \n";



#include "petscsys.h"
#include "petscvec.h"
#include "petscdmda.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode      ierr;
  Vec                 x,y;

  PetscReal           *values;

  PetscViewer         viewer_in,viewer_out;
  DM                  daf,dac;
  Vec            scaling;
  Mat            interp;
  PetscInt            i;
  
  PetscInitialize(&argc,&argv, (char*)0, help);
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,1024,1024,PETSC_DECIDE,PETSC_DECIDE, 1, 1,PETSC_NULL,PETSC_NULL,&daf);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(daf,&x);CHKERRQ(ierr);
  ierr = VecGetArray(x,&values);CHKERRQ(ierr);

  ierr = DMCoarsen(daf,PETSC_COMM_WORLD,&dac);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dac,&y);CHKERRQ(ierr);
  ierr = DMGetInterpolation(dac,daf,&interp,&scaling);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi",FILE_MODE_READ,&viewer_in);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer_in,values,104876,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = MatRestrict(interp,x,y);
  ierr = VecPointwiseMult(y,y,scaling);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi2",FILE_MODE_WRITE,&viewer_out);CHKERRQ(ierr);
  ierr = VecView(y,viewer_out);CHKERRQ(ierr);
  ierr = VecView(x,viewer_out);CHKERRQ(ierr);
  
  ierr = PetscViewerDestroy(&viewer_in);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_out);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
