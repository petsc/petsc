static const char help[] = "Test DMCreateInjection() for mapping coordinates in 3D";

#include <petscvec.h>
#include <petscmat.h>
#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode test1_DAInjection3d(PetscInt mx, PetscInt my, PetscInt mz)
{
  PetscErrorCode   ierr;
  DM               dac,daf;
  PetscViewer      vv;
  Vec              ac,af;
  PetscInt         periodicity;
  DMBoundaryType   bx,by,bz;

  PetscFunctionBeginUser;
  bx = DM_BOUNDARY_NONE;
  by = DM_BOUNDARY_NONE;
  bz = DM_BOUNDARY_NONE;

  periodicity = 0;

  ierr = PetscOptionsGetInt(NULL,NULL,"-periodic", &periodicity, NULL);CHKERRQ(ierr);
  if (periodicity==1) {
    bx = DM_BOUNDARY_PERIODIC;
  } else if (periodicity==2) {
    by = DM_BOUNDARY_PERIODIC;
  } else if (periodicity==3) {
    bz = DM_BOUNDARY_PERIODIC;
  }

  ierr = DMDACreate3d(PETSC_COMM_WORLD, bx,by,bz, DMDA_STENCIL_BOX,mx+1, my+1,mz+1,PETSC_DECIDE, PETSC_DECIDE,PETSC_DECIDE,1, /* 1 dof */
                      1, /* stencil = 1 */NULL,NULL,NULL,&daf);CHKERRQ(ierr);
  ierr = DMSetFromOptions(daf);CHKERRQ(ierr);
  ierr = DMSetUp(daf);CHKERRQ(ierr);

  ierr = DMCoarsen(daf,MPI_COMM_NULL,&dac);CHKERRQ(ierr);

  ierr = DMDASetUniformCoordinates(dac, -1.0,1.0, -1.0,1.0, -1.0,1.0);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(daf, -1.0,1.0, -1.0,1.0, -1.0,1.0);CHKERRQ(ierr);

  {
    DM         cdaf,cdac;
    Vec        coordsc,coordsf,coordsf2;
    Mat        inject;
    VecScatter vscat;
    Mat        interp;
    PetscReal  norm;

    ierr = DMGetCoordinateDM(dac,&cdac);CHKERRQ(ierr);
    ierr = DMGetCoordinateDM(daf,&cdaf);CHKERRQ(ierr);

    ierr = DMGetCoordinates(dac,&coordsc);CHKERRQ(ierr);
    ierr = DMGetCoordinates(daf,&coordsf);CHKERRQ(ierr);

    ierr = DMCreateInjection(cdac,cdaf,&inject);CHKERRQ(ierr);
    ierr = MatScatterGetVecScatter(inject,&vscat);CHKERRQ(ierr);
    ierr = VecScatterBegin(vscat,coordsf,coordsc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = VecScatterEnd(vscat  ,coordsf,coordsc,INSERT_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    ierr = MatDestroy(&inject);CHKERRQ(ierr);

    ierr = DMCreateInterpolation(cdac,cdaf,&interp,NULL);CHKERRQ(ierr);
    ierr = VecDuplicate(coordsf,&coordsf2);CHKERRQ(ierr);
    ierr = MatInterpolate(interp,coordsc,coordsf2);CHKERRQ(ierr);
    ierr = VecAXPY(coordsf2,-1.0,coordsf);CHKERRQ(ierr);
    ierr = VecNorm(coordsf2,NORM_MAX,&norm);CHKERRQ(ierr);
    /* The fine coordinates are only reproduced in certain cases */
    if (!bx && !by && !bz && norm > PETSC_SQRT_MACHINE_EPSILON) {ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm %g\n",(double)norm);CHKERRQ(ierr);}
    ierr = VecDestroy(&coordsf2);CHKERRQ(ierr);
    ierr = MatDestroy(&interp);CHKERRQ(ierr);
  }

  if (0) {
    ierr = DMCreateGlobalVector(dac,&ac);CHKERRQ(ierr);
    ierr = VecZeroEntries(ac);CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(daf,&af);CHKERRQ(ierr);
    ierr = VecZeroEntries(af);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "dac_7.vtk", &vv);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(vv, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(dac, vv);CHKERRQ(ierr);
    ierr = VecView(ac, vv);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(vv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vv);CHKERRQ(ierr);

    ierr = PetscViewerASCIIOpen(PETSC_COMM_WORLD, "daf_7.vtk", &vv);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(vv, PETSC_VIEWER_ASCII_VTK);CHKERRQ(ierr);
    ierr = DMView(daf, vv);CHKERRQ(ierr);
    ierr = VecView(af, vv);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(vv);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&vv);CHKERRQ(ierr);
    ierr = VecDestroy(&ac);CHKERRQ(ierr);
    ierr = VecDestroy(&af);CHKERRQ(ierr);
  }
  ierr = DMDestroy(&dac);CHKERRQ(ierr);
  ierr = DMDestroy(&daf);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       mx,my,mz;

  ierr = PetscInitialize(&argc,&argv,0,help);if (ierr) return ierr;
  mx   = 2;
  my   = 2;
  mz   = 2;
  ierr = PetscOptionsGetInt(NULL,NULL,"-mx", &mx, 0);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-my", &my, 0);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-mz", &mz, 0);CHKERRQ(ierr);
  ierr = test1_DAInjection3d(mx,my,mz);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

      test:
         nsize: 5
         args: -mx 30 -my 30 -mz 30 -periodic 0 -da_processors_x 5

      test:
         suffix: 2
         nsize: 5
         args: -mx 29 -my 30 -mz 30 -periodic 1 -da_processors_x 5

      test:
         suffix: 3
         nsize: 5
         args: -mx 30 -my 29 -mz 30 -periodic 2 -da_processors_x 5

      test:
         suffix: 4
         nsize: 5
         args: -mx 30 -my 30 -mz 29 -periodic 3 -da_processors_x 5

      test:
         suffix: 5
         nsize: 5
         args: -mx 30 -my 30 -mz 30 -periodic 0 -da_processors_y 5

      test:
         suffix: 6
         nsize: 5
         args: -mx 29 -my 30 -mz 30 -periodic 1 -da_processors_y 5

      test:
         suffix: 7
         nsize: 5
         args: -mx 30 -my 29 -mz 30 -periodic 2 -da_processors_y 5

      test:
         suffix: 8
         nsize: 5
         args: -mx 30 -my 30 -mz 29 -periodic 3 -da_processors_y 5

      test:
         suffix: 9
         nsize: 5
         args: -mx 30 -my 30 -mz 30 -periodic 0 -da_processors_z 5

      test:
         suffix: 10
         nsize: 5
         args: -mx 29 -my 30 -mz 30 -periodic 1 -da_processors_z 5

      test:
         suffix: 11
         nsize: 5
         args: -mx 30 -my 29 -mz 30 -periodic 2 -da_processors_z 5

      test:
         suffix: 12
         nsize: 5
         args: -mx 30 -my 30 -mz 29 -periodic 3 -da_processors_z 5

      test:
         suffix: 13
         nsize: 5
         args: -mx 30 -my 30 -mz 30 -periodic 0

      test:
         suffix: 14
         nsize: 5
         args: -mx 29 -my 30 -mz 30 -periodic 1

      test:
         suffix: 15
         nsize: 5
         args: -mx 30 -my 29 -mz 30 -periodic 2

      test:
         suffix: 16
         nsize: 5
         args: -mx 30 -my 30 -mz 29 -periodic 3

TEST*/
