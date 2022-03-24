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

  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-periodic", &periodicity, NULL));
  if (periodicity==1) {
    bx = DM_BOUNDARY_PERIODIC;
  } else if (periodicity==2) {
    by = DM_BOUNDARY_PERIODIC;
  } else if (periodicity==3) {
    bz = DM_BOUNDARY_PERIODIC;
  }

  ierr = DMDACreate3d(PETSC_COMM_WORLD, bx,by,bz, DMDA_STENCIL_BOX,mx+1, my+1,mz+1,PETSC_DECIDE, PETSC_DECIDE,PETSC_DECIDE,1, /* 1 dof */
                      1, /* stencil = 1 */NULL,NULL,NULL,&daf);CHKERRQ(ierr);
  CHKERRQ(DMSetFromOptions(daf));
  CHKERRQ(DMSetUp(daf));

  CHKERRQ(DMCoarsen(daf,MPI_COMM_NULL,&dac));

  CHKERRQ(DMDASetUniformCoordinates(dac, -1.0,1.0, -1.0,1.0, -1.0,1.0));
  CHKERRQ(DMDASetUniformCoordinates(daf, -1.0,1.0, -1.0,1.0, -1.0,1.0));

  {
    DM         cdaf,cdac;
    Vec        coordsc,coordsf,coordsf2;
    Mat        inject;
    VecScatter vscat;
    Mat        interp;
    PetscReal  norm;

    CHKERRQ(DMGetCoordinateDM(dac,&cdac));
    CHKERRQ(DMGetCoordinateDM(daf,&cdaf));

    CHKERRQ(DMGetCoordinates(dac,&coordsc));
    CHKERRQ(DMGetCoordinates(daf,&coordsf));

    CHKERRQ(DMCreateInjection(cdac,cdaf,&inject));
    CHKERRQ(MatScatterGetVecScatter(inject,&vscat));
    CHKERRQ(VecScatterBegin(vscat,coordsf,coordsc,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(VecScatterEnd(vscat  ,coordsf,coordsc,INSERT_VALUES,SCATTER_FORWARD));
    CHKERRQ(MatDestroy(&inject));

    CHKERRQ(DMCreateInterpolation(cdac,cdaf,&interp,NULL));
    CHKERRQ(VecDuplicate(coordsf,&coordsf2));
    CHKERRQ(MatInterpolate(interp,coordsc,coordsf2));
    CHKERRQ(VecAXPY(coordsf2,-1.0,coordsf));
    CHKERRQ(VecNorm(coordsf2,NORM_MAX,&norm));
    /* The fine coordinates are only reproduced in certain cases */
    if (!bx && !by && !bz && norm > PETSC_SQRT_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"Norm %g\n",(double)norm));
    CHKERRQ(VecDestroy(&coordsf2));
    CHKERRQ(MatDestroy(&interp));
  }

  if (0) {
    CHKERRQ(DMCreateGlobalVector(dac,&ac));
    CHKERRQ(VecZeroEntries(ac));

    CHKERRQ(DMCreateGlobalVector(daf,&af));
    CHKERRQ(VecZeroEntries(af));

    CHKERRQ(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "dac_7.vtu", &vv));
    CHKERRQ(VecView(ac, vv));
    CHKERRQ(PetscViewerDestroy(&vv));

    CHKERRQ(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "daf_7.vtu", &vv));
    CHKERRQ(VecView(af, vv));
    CHKERRQ(PetscViewerDestroy(&vv));
    CHKERRQ(VecDestroy(&ac));
    CHKERRQ(VecDestroy(&af));
  }
  CHKERRQ(DMDestroy(&dac));
  CHKERRQ(DMDestroy(&daf));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscInt       mx,my,mz;

  CHKERRQ(PetscInitialize(&argc,&argv,0,help));
  mx   = 2;
  my   = 2;
  mz   = 2;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mx", &mx, 0));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-my", &my, 0));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-mz", &mz, 0));
  CHKERRQ(test1_DAInjection3d(mx,my,mz));
  CHKERRQ(PetscFinalize());
  return 0;
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
