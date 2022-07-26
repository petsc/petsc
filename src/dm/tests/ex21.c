static const char help[] = "Test DMCreateInjection() for mapping coordinates in 3D";

#include <petscvec.h>
#include <petscmat.h>
#include <petscdm.h>
#include <petscdmda.h>

PetscErrorCode test1_DAInjection3d(PetscInt mx, PetscInt my, PetscInt mz)
{
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

  PetscCall(PetscOptionsGetInt(NULL,NULL,"-periodic", &periodicity, NULL));
  if (periodicity==1) {
    bx = DM_BOUNDARY_PERIODIC;
  } else if (periodicity==2) {
    by = DM_BOUNDARY_PERIODIC;
  } else if (periodicity==3) {
    bz = DM_BOUNDARY_PERIODIC;
  }

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, bx,by,bz, DMDA_STENCIL_BOX,mx+1, my+1,mz+1,PETSC_DECIDE, PETSC_DECIDE,PETSC_DECIDE,1, /* 1 dof */
                         1, /* stencil = 1 */NULL,NULL,NULL,&daf));
  PetscCall(DMSetFromOptions(daf));
  PetscCall(DMSetUp(daf));

  PetscCall(DMCoarsen(daf,MPI_COMM_NULL,&dac));

  PetscCall(DMDASetUniformCoordinates(dac, -1.0,1.0, -1.0,1.0, -1.0,1.0));
  PetscCall(DMDASetUniformCoordinates(daf, -1.0,1.0, -1.0,1.0, -1.0,1.0));

  {
    DM         cdaf,cdac;
    Vec        coordsc,coordsf,coordsf2;
    Mat        inject;
    VecScatter vscat;
    Mat        interp;
    PetscReal  norm;

    PetscCall(DMGetCoordinateDM(dac,&cdac));
    PetscCall(DMGetCoordinateDM(daf,&cdaf));

    PetscCall(DMGetCoordinates(dac,&coordsc));
    PetscCall(DMGetCoordinates(daf,&coordsf));

    PetscCall(DMCreateInjection(cdac,cdaf,&inject));
    PetscCall(MatScatterGetVecScatter(inject,&vscat));
    PetscCall(VecScatterBegin(vscat,coordsf,coordsc,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(VecScatterEnd(vscat  ,coordsf,coordsc,INSERT_VALUES,SCATTER_FORWARD));
    PetscCall(MatDestroy(&inject));

    PetscCall(DMCreateInterpolation(cdac,cdaf,&interp,NULL));
    PetscCall(VecDuplicate(coordsf,&coordsf2));
    PetscCall(MatInterpolate(interp,coordsc,coordsf2));
    PetscCall(VecAXPY(coordsf2,-1.0,coordsf));
    PetscCall(VecNorm(coordsf2,NORM_MAX,&norm));
    /* The fine coordinates are only reproduced in certain cases */
    if (!bx && !by && !bz && norm > PETSC_SQRT_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Norm %g\n",(double)norm));
    PetscCall(VecDestroy(&coordsf2));
    PetscCall(MatDestroy(&interp));
  }

  if (0) {
    PetscCall(DMCreateGlobalVector(dac,&ac));
    PetscCall(VecZeroEntries(ac));

    PetscCall(DMCreateGlobalVector(daf,&af));
    PetscCall(VecZeroEntries(af));

    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "dac_7.vtu", &vv));
    PetscCall(VecView(ac, vv));
    PetscCall(PetscViewerDestroy(&vv));

    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_WORLD, "daf_7.vtu", &vv));
    PetscCall(VecView(af, vv));
    PetscCall(PetscViewerDestroy(&vv));
    PetscCall(VecDestroy(&ac));
    PetscCall(VecDestroy(&af));
  }
  PetscCall(DMDestroy(&dac));
  PetscCall(DMDestroy(&daf));
  PetscFunctionReturn(0);
}

int main(int argc,char **argv)
{
  PetscInt       mx,my,mz;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,0,help));
  mx   = 2;
  my   = 2;
  mz   = 2;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mx", &mx, 0));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-my", &my, 0));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-mz", &mz, 0));
  PetscCall(test1_DAInjection3d(mx,my,mz));
  PetscCall(PetscFinalize());
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
