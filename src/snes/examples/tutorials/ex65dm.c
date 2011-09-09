static char help[] = "coarsening with DM.\n";



#include "petscsys.h"
#include "petscvec.h"
#include "petscdmda.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode      ierr;
  Vec                 x,y1,y2,y3;
  PetscReal           *values;
  PetscViewer         viewer_in,viewer_out1,viewer_out2,viewer_out3;
  DM                  daf,dac1,dac2,dac3;
  Vec                 scaling1,scaling2,scaling3;
  Mat                 interp1,interp2,interp3;
  PetscInt            i;
  
  PetscInitialize(&argc,&argv, (char*)0, help);
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,1024,1024,PETSC_DECIDE,PETSC_DECIDE, 1, 1,PETSC_NULL,PETSC_NULL,&daf);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(daf,&x);CHKERRQ(ierr);
  ierr = VecGetArray(x,&values);CHKERRQ(ierr);

  ierr = DMCoarsen(daf,PETSC_COMM_WORLD,&dac1);CHKERRQ(ierr);
  ierr = DMCoarsen(dac1,PETSC_COMM_WORLD,&dac2);CHKERRQ(ierr);
  ierr = DMCoarsen(dac2,PETSC_COMM_WORLD,&dac3);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dac1,&y1);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dac2,&y2);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dac3,&y3);CHKERRQ(ierr);
  ierr = DMGetInterpolation(dac1,daf,&interp1,&scaling1);CHKERRQ(ierr);
  ierr = DMGetInterpolation(dac2,dac1,&interp2,&scaling2);CHKERRQ(ierr);
  ierr = DMGetInterpolation(dac3,dac2,&interp3,&scaling3);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi",FILE_MODE_READ,&viewer_in);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer_in,values,1048576,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = MatRestrict(interp1,x,y1);
  ierr = VecPointwiseMult(y1,y1,scaling1);CHKERRQ(ierr);
  ierr = MatRestrict(interp2,y1,y2);
  ierr = VecPointwiseMult(y2,y2,scaling2);CHKERRQ(ierr);
  ierr = MatRestrict(interp3,y2,y3);
  ierr = VecPointwiseMult(y3,y3,scaling3);CHKERRQ(ierr);
  
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi1",FILE_MODE_WRITE,&viewer_out1);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi2",FILE_MODE_WRITE,&viewer_out2);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi3",FILE_MODE_WRITE,&viewer_out3);CHKERRQ(ierr);
  ierr = VecView(y1,viewer_out1);CHKERRQ(ierr);
  ierr = VecView(x,viewer_out1);CHKERRQ(ierr);
  ierr = VecView(y2,viewer_out2);CHKERRQ(ierr);
  ierr = VecView(y3,viewer_out3);CHKERRQ(ierr);
  
  ierr = PetscViewerDestroy(&viewer_in);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_out1);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y1);CHKERRQ(ierr);
  ierr = VecDestroy(&y2);CHKERRQ(ierr);
  ierr = VecDestroy(&y3);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
