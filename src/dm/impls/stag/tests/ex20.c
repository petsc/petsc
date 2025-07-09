static char help[] = "Test DMStag transfer operators.\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc, char **argv)
{
  DM        dmc, dmf;
  PetscInt  dim;
  PetscBool flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Supply -dim option");
  if (dim == 1) PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 3, 1, 1, DMSTAG_STENCIL_BOX, 1, NULL, &dmc));
  else if (dim == 2) PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 3, 3, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 1, DMSTAG_STENCIL_BOX, 1, NULL, NULL, &dmc));
  else if (dim == 3) PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 3, 3, 3, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 1, 1, DMSTAG_STENCIL_BOX, 1, NULL, NULL, NULL, &dmc));
  else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "dim must be 1, 2, or 3");
  PetscCall(DMSetFromOptions(dmc));
  PetscCall(DMSetUp(dmc));

  /* Directly create a coarsened DM and transfer operators */
  PetscCall(DMRefine(dmc, MPI_COMM_NULL, &dmf));
  {
    Mat       Ai;
    Vec       vc, vf;
    PetscInt  size;
    PetscReal norm;

    PetscCall(DMCreateInterpolation(dmc, dmf, &Ai, NULL));
    PetscCall(MatCreateVecs(Ai, &vc, &vf));
    PetscCall(VecSet(vc, 1.0));
    PetscCall(MatMult(Ai, vc, vf));
    PetscCall(VecGetSize(vf, &size));
    PetscCall(VecNorm(vf, NORM_1, &norm));
    PetscCheck((norm - size) / (PetscReal)size <= PETSC_MACHINE_EPSILON * 10.0, PetscObjectComm((PetscObject)dmc), PETSC_ERR_PLIB, "Numerical test failed");
    PetscCall(MatDestroy(&Ai));
    PetscCall(VecDestroy(&vc));
    PetscCall(VecDestroy(&vf));
  }
  {
    Mat       Ar;
    Vec       vf, vc;
    PetscInt  size;
    PetscReal norm;

    PetscCall(DMCreateRestriction(dmc, dmf, &Ar));
    PetscCall(MatCreateVecs(Ar, &vf, &vc));
    PetscCall(VecSet(vf, 1.0));
    PetscCall(MatMult(Ar, vf, vc));
    PetscCall(VecGetSize(vc, &size));
    PetscCall(VecNorm(vc, NORM_1, &norm));
    PetscCheck((norm - size) / (PetscReal)size <= PETSC_MACHINE_EPSILON * 10.0, PetscObjectComm((PetscObject)dmc), PETSC_ERR_PLIB, "Numerical test failed");
    PetscCall(MatDestroy(&Ar));
    PetscCall(VecDestroy(&vf));
    PetscCall(VecDestroy(&vc));
  }
  PetscCall(DMDestroy(&dmf));

  PetscCall(DMDestroy(&dmc));
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
      suffix: 1d_ratio
      nsize: 1
      args: -dim 1 -stag_refine_x 3
      output_file: output/empty.out

   test:
      suffix: 1d_par
      nsize: 2
      args: -dim 1 -stag_grid_x 6
      output_file: output/empty.out

   test:
      suffix: 2d
      nsize: 1
      args: -dim 2
      output_file: output/empty.out

   test:
      suffix: 2d_ratio
      nsize: 1
      args: -dim 2 -stag_refine_x 3 -stag_refine_y 4
      output_file: output/empty.out

   test:
      suffix: 2d_par
      nsize: 4
      args: -dim 2 -stag_grid_x 6 -stag_grid_y 7
      output_file: output/empty.out

   test:
      suffix: 3d
      nsize: 1
      args: -dim 3
      output_file: output/empty.out

   test:
      suffix: 3d_ratio
      nsize: 1
      args: -dim 3 -stag_refine_x 3 -stag_refine_y 4 -stag_refine_z 5
      output_file: output/empty.out

   test:
      suffix: 3d_par
      nsize: 8
      args: -dim 3 -stag_grid_x 6 -stag_grid_y 7 -stag_grid_z 8
      output_file: output/empty.out

TEST*/
