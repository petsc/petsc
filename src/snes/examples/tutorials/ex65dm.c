static char help[] = "coarsening with DM.\n";



#include "petscsys.h"
#include "petscvec.h"
#include "petscdmda.h"


#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc, char **argv)
{
  PetscErrorCode      ierr;
  Vec                 x,yp1,yp2,yp3,yp4,ym1,ym2,ym3,ym4;
  PetscReal           *values;
  PetscViewer         viewer_in,viewer_outp1,viewer_outp2,viewer_outp3,viewer_outp4;
  PetscViewer         viewer_outm1,viewer_outm2,viewer_outm3,viewer_outm4;
  DM                  daf,dac1,dac2,dac3,dac4,daf1,daf2,daf3,daf4;
  Vec                 scaling_p1,scaling_p2,scaling_p3,scaling_p4;
  Mat                 interp_p1,interp_p2,interp_p3,interp_p4,interp_m1,interp_m2,interp_m3,interp_m4;
  PetscInt            i;
  
  PetscInitialize(&argc,&argv, (char*)0, help);
  ierr = DMDACreate2d(PETSC_COMM_WORLD,DMDA_BOUNDARY_PERIODIC,DMDA_BOUNDARY_PERIODIC,DMDA_STENCIL_BOX,1024,1024,PETSC_DECIDE,PETSC_DECIDE, 1, 1,PETSC_NULL,PETSC_NULL,&daf);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(daf,&x);CHKERRQ(ierr);
  ierr = VecGetArray(x,&values);CHKERRQ(ierr);

  ierr = DMCoarsen(daf,PETSC_COMM_WORLD,&dac1); CHKERRQ(ierr);
  ierr = DMCoarsen(dac1,PETSC_COMM_WORLD,&dac2);CHKERRQ(ierr);
  ierr = DMCoarsen(dac2,PETSC_COMM_WORLD,&dac3);CHKERRQ(ierr);
  ierr = DMCoarsen(dac3,PETSC_COMM_WORLD,&dac4);CHKERRQ(ierr);
  ierr = DMRefine(daf,PETSC_COMM_WORLD,&daf1);  CHKERRQ(ierr);
  ierr = DMRefine(daf1,PETSC_COMM_WORLD,&daf2);CHKERRQ(ierr);
  ierr = DMRefine(daf2,PETSC_COMM_WORLD,&daf3);CHKERRQ(ierr);
  ierr = DMRefine(daf3,PETSC_COMM_WORLD,&daf4);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(dac1,&yp1);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dac2,&yp2);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dac3,&yp3);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(dac4,&yp4);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(daf1,&ym1);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(daf2,&ym2);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(daf3,&ym3);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(daf4,&ym4);CHKERRQ(ierr);

  ierr = DMCreateInterpolation(dac1,daf,&interp_p1,&scaling_p1);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(dac2,dac1,&interp_p2,&scaling_p2);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(dac3,dac2,&interp_p3,&scaling_p3);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(dac4,dac3,&interp_p4,&scaling_p4);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(daf,daf1,&interp_m1,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(daf1,daf2,&interp_m2,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(daf2,daf3,&interp_m3,PETSC_NULL);CHKERRQ(ierr);
  ierr = DMCreateInterpolation(daf3,daf4,&interp_m4,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi",FILE_MODE_READ,&viewer_in);CHKERRQ(ierr);
  ierr = PetscViewerBinaryRead(viewer_in,values,1048576,PETSC_DOUBLE);CHKERRQ(ierr);
  ierr = MatRestrict(interp_p1,x,yp1);
  ierr = VecPointwiseMult(yp1,yp1,scaling_p1);CHKERRQ(ierr);
  ierr = MatRestrict(interp_p2,yp1,yp2);
  ierr = VecPointwiseMult(yp2,yp2,scaling_p2);CHKERRQ(ierr);
  ierr = MatRestrict(interp_p3,yp2,yp3);
  ierr = VecPointwiseMult(yp3,yp3,scaling_p3);CHKERRQ(ierr);
  ierr = MatRestrict(interp_p4,yp3,yp4);
  ierr = VecPointwiseMult(yp4,yp4,scaling_p4);CHKERRQ(ierr);
  ierr = MatRestrict(interp_m1,x,ym1);
  ierr = MatRestrict(interp_m2,ym1,ym2);
  ierr = MatRestrict(interp_m3,ym2,ym3);
  ierr = MatRestrict(interp_m4,ym3,ym4);
  
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi1",FILE_MODE_WRITE,&viewer_outp1);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi2",FILE_MODE_WRITE,&viewer_outp2);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi3",FILE_MODE_WRITE,&viewer_outp3);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phi4",FILE_MODE_WRITE,&viewer_outp4);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phim1",FILE_MODE_WRITE,&viewer_outm1);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phim2",FILE_MODE_WRITE,&viewer_outm2);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phim3",FILE_MODE_WRITE,&viewer_outm3);CHKERRQ(ierr);
  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"phim4",FILE_MODE_WRITE,&viewer_outm4);CHKERRQ(ierr);

  ierr = VecView(yp1,viewer_outp1);CHKERRQ(ierr);
  ierr = VecView(x,viewer_outp1);CHKERRQ(ierr);
  ierr = VecView(yp2,viewer_outp2);CHKERRQ(ierr);
  ierr = VecView(yp3,viewer_outp3);CHKERRQ(ierr);
  ierr = VecView(yp4,viewer_outp4);CHKERRQ(ierr);
  ierr = VecView(ym1,viewer_outm1);CHKERRQ(ierr);
  ierr = VecView(ym2,viewer_outm2);CHKERRQ(ierr);
  ierr = VecView(ym3,viewer_outm3);CHKERRQ(ierr);
  ierr = VecView(ym4,viewer_outm4);CHKERRQ(ierr);
  
  ierr = PetscViewerDestroy(&viewer_in);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_outp1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_outp2);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_outp3);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_outp4);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer_outm1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_outm2);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_outm3);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer_outm4);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&yp1);CHKERRQ(ierr);
  ierr = VecDestroy(&yp2);CHKERRQ(ierr);
  ierr = VecDestroy(&yp3);CHKERRQ(ierr);
  ierr = VecDestroy(&yp4);CHKERRQ(ierr);
  ierr = VecDestroy(&ym1);CHKERRQ(ierr);
  ierr = VecDestroy(&ym2);CHKERRQ(ierr);
  ierr = VecDestroy(&ym3);CHKERRQ(ierr);
  ierr = VecDestroy(&ym4);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
