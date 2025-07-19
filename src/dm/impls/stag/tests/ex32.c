static char help[] = "Test DMStagRestrictSimple()\n\n";

#include <petscdmstag.h>

int main(int argc, char **argv)
{
  DM        dmf, dmc;
  Vec       gf, gc, lf, lc;
  PetscInt  dim, size;
  PetscReal norm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  dim = 2;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  switch (dim) {
  case 1:
    PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 4, 2, 3, DMSTAG_STENCIL_BOX, 1, NULL, &dmc));
    break;
  case 2:
    PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 8, PETSC_DECIDE, PETSC_DECIDE, 2, 3, 4, DMSTAG_STENCIL_BOX, 1, NULL, NULL, &dmc));
    break;
  case 3:
    PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 2, 4, 6, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 2, 3, 4, 3, DMSTAG_STENCIL_BOX, 1, NULL, NULL, NULL, &dmc));
    break;
  default:
    SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "Not Implemented!");
  }
  PetscCall(DMSetFromOptions(dmc));
  PetscCall(DMSetUp(dmc));
  PetscCall(DMRefine(dmc, MPI_COMM_NULL, &dmf));

  PetscCall(DMCreateGlobalVector(dmf, &gf));
  PetscCall(VecSet(gf, 1.0));
  PetscCall(DMCreateLocalVector(dmf, &lf));
  PetscCall(DMGlobalToLocal(dmf, gf, INSERT_VALUES, lf));

  PetscCall(DMCreateGlobalVector(dmc, &gc));
  PetscCall(DMCreateLocalVector(dmc, &lc));

  PetscCall(DMStagRestrictSimple(dmf, lf, dmc, lc));

  PetscCall(DMLocalToGlobal(dmc, lc, INSERT_VALUES, gc));

  PetscCall(VecGetSize(gc, &size));
  PetscCall(VecNorm(gc, NORM_1, &norm));
  PetscCheck((norm - size) / (PetscReal)size <= PETSC_MACHINE_EPSILON * 10.0, PetscObjectComm((PetscObject)dmc), PETSC_ERR_PLIB, "Numerical test failed");
  PetscCall(VecDestroy(&gc));
  PetscCall(VecDestroy(&gf));
  PetscCall(VecDestroy(&lc));
  PetscCall(VecDestroy(&lf));
  PetscCall(DMDestroy(&dmc));
  PetscCall(DMDestroy(&dmf));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1d
      nsize: 1
      args: -dim 1
      output_file: output/empty.out

   test:
      suffix: 1d_par
      nsize: 4
      args: -dim 1
      output_file: output/empty.out

   test:
      suffix: 1d_ratio
      nsize: 1
      args: -dim 1 -stag_refine_x 3
      output_file: output/empty.out

   test:
      suffix: 2d
      nsize: 1
      args: -dim 2
      output_file: output/empty.out

   test:
      suffix: 2d_par
      nsize: 2
      args: -dim 2
      output_file: output/empty.out

   test:
      suffix: 2d_par_2
      nsize: 8
      args: -dim 2
      output_file: output/empty.out

   test:
      suffix: 2d_ratio
      nsize: 1
      args: -dim 2 -stag_refine_x 3 -stag_refine_y 4
      output_file: output/empty.out

   test:
      suffix: 3d
      nsize: 1
      args: -dim 3
      output_file: output/empty.out

   test:
      suffix: 3d_par
      nsize: 2
      args: -dim 3
      output_file: output/empty.out

   test:
      suffix: 3d_ratio
      nsize: 1
      args: -dim 3 -stag_refine_x 3 -stag_refine_y 4 -stag_refine_z 5
      output_file: output/empty.out
TEST*/
