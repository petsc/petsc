static const char help[] = "Test DMCreateInjection() for mapping coordinates in 3D";

#include <petscvec.h>
#include <petscmat.h>
#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "test1_DAInjection3d"
PetscErrorCode test1_DAInjection3d( PetscInt mx, PetscInt my, PetscInt mz )
{
  PetscErrorCode   ierr;
  DM               dac,daf;
  PetscViewer      vv;
  Vec              ac,af;
  PetscInt         periodicity;
  DMDABoundaryType bx,by,bz;

  PetscFunctionBegin;
  bx = DMDA_BOUNDARY_NONE;
  by = DMDA_BOUNDARY_NONE;
  bz = DMDA_BOUNDARY_NONE;
  periodicity = 0;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-periodic", &periodicity, PETSC_NULL);CHKERRQ(ierr);
  if (periodicity==1) {
    bx = DMDA_BOUNDARY_PERIODIC;
  } else if (periodicity==2) {
    by = DMDA_BOUNDARY_PERIODIC;
  } else if (periodicity==3) {
    bz = DMDA_BOUNDARY_PERIODIC;
  }

  ierr = DMDACreate3d(PETSC_COMM_WORLD, bx,by,bz, DMDA_STENCIL_BOX,
                      mx+1, my+1,mz+1,
                      PETSC_DECIDE, PETSC_DECIDE,PETSC_DECIDE,
                      1, /* 1 dof */
                      1, /* stencil = 1 */
                      PETSC_NULL,PETSC_NULL,PETSC_NULL,
                      &daf); CHKERRQ(ierr);

  ierr = DMSetFromOptions(daf);CHKERRQ(ierr);

  ierr = DMCoarsen(daf,MPI_COMM_NULL,&dac);CHKERRQ(ierr);

  ierr = DMDASetUniformCoordinates(dac, -1.0,1.0, -1.0,1.0, -1.0,1.0 );CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(daf, -1.0,1.0, -1.0,1.0, -1.0,1.0 );CHKERRQ(ierr);

  {
    DM         cdaf,cdac;
    Vec        coordsc,coordsf,coordsf2;
    VecScatter inject;
    Mat        interp;
    PetscReal  norm;

    ierr = DMGetCoordinateDM(dac,&cdac);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(daf,&cdaf);CHKERRQ(ierr);

    ierr = DMGetCoordinates(dac,&coordsc);CHKERRQ(ierr);
    ierr = DMGetCoordinates(daf,&coordsf);CHKERRQ(ierr);

    ierr = DMCreateInjection(cdac,cdaf,&inject);CHKERRQ(ierr);

    ierr = VecScatterBegin(inject,coordsf,coordsc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(inject  ,coordsf,coordsc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterDestroy(&inject);CHKERRQ(ierr);

    ierr = DMCreateInterpolation(cdac,cdaf,&interp,PETSC_NULL);CHKERRQ(ierr);
    ierr = VecDuplicate(coordsf,&coordsf2);CHKERRQ(ierr);
    ierr = MatInterpolate(interp,coordsc,coordsf2);CHKERRQ(ierr);
    ierr = VecAXPY(coordsf2,-1.0,coordsf);CHKERRQ(ierr);
    ierr = VecNorm(coordsf2,NORM_MAX,&norm);CHKERRQ(ierr);
    /* The fine coordinates are only reproduced in certain cases */
    if (!bx && !by && !bz && norm > 1.e-10) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm %G\n",norm);CHKERRQ(ierr);}
    ierr = VecDestroy(&coordsf2);CHKERRQ(ierr);
    ierr = MatDestroy(&interp);CHKERRQ(ierr);
  }

  if (0) {
    ierr = DMCreateGlobalVector(dac,&ac); CHKERRQ(ierr);
    ierr = VecZeroEntries(ac);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(daf,&af); CHKERRQ(ierr);
    ierr = VecZeroEntries(af);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "dac_7.vtk", &vv);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(vv, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(dac, vv);CHKERRQ(ierr);
    ierr = VecView(ac, vv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vv);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "daf_7.vtk", &vv);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(vv, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(daf, vv);CHKERRQ(ierr);
    ierr = VecView(af, vv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vv);CHKERRQ(ierr);
    ierr = VecDestroy(&ac);CHKERRQ(ierr);
    ierr = VecDestroy(&af);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dac);CHKERRQ(ierr);
  ierr = DMDestroy(&daf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt mx,my,mz;

  ierr = PetscInitialize(&argc,&argv,0,help);
  mx = 2;
  my = 2;
  mz = 2;
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mx", &mx, 0 );CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-my", &my, 0 );CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-mz", &mz, 0 );CHKERRQ(ierr);

  ierr = test1_DAInjection3d(mx,my,mz);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
