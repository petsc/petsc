static char help[] = "Test local-to-local for DMStag.\n\n";

#include <petscdmstag.h>

int main(int argc, char **argv)
{
  DM          dm;
  PetscInt    dim, start, end, i;
  PetscBool   flg;
  Vec         g, l1, l2;
  PetscMPIInt rank;
  PetscScalar value;
  PetscReal   norm, work;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "Supply -dim option");

  if (dim == 1) PetscCall(DMStagCreate1d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, 64, 1, 1, DMSTAG_STENCIL_BOX, 1, NULL, &dm));
  else if (dim == 2) PetscCall(DMStagCreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 8, 8, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 1, DMSTAG_STENCIL_BOX, 1, NULL, NULL, &dm));
  else if (dim == 3) PetscCall(DMStagCreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, 4, 4, 4, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, 1, 1, 1, 1, DMSTAG_STENCIL_BOX, 1, NULL, NULL, NULL, &dm));
  else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_SUP, "dim must be 1, 2, or 3");
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));

  PetscCall(DMCreateGlobalVector(dm, &g));
  PetscCall(DMCreateLocalVector(dm, &l1));
  PetscCall(VecDuplicate(l1, &l2));

  PetscCall(VecSet(l1, 0.0));
  PetscCall(VecSet(l2, 0.0));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCall(VecGetOwnershipRange(g, &start, &end));
  for (i = start; i < end; ++i) {
    value = rank + i;
    PetscCall(VecSetValues(g, 1, &i, &value, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(g));
  PetscCall(VecAssemblyEnd(g));

  PetscCall(DMGlobalToLocalBegin(dm, g, INSERT_VALUES, l1));
  PetscCall(DMGlobalToLocalEnd(dm, g, INSERT_VALUES, l1));

  PetscCall(DMLocalToLocalBegin(dm, l1, INSERT_VALUES, l2));
  PetscCall(DMLocalToLocalEnd(dm, l1, INSERT_VALUES, l2));

  /* l1 and l2 must be same. */
  PetscCall(VecAXPY(l2, -1.0, l1));
  PetscCall(VecNorm(l2, NORM_MAX, &work));
  PetscCallMPI(MPIU_Allreduce(&work, &norm, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "norm = %g\n", (double)norm));

  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&l1));
  PetscCall(VecDestroy(&l2));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      nsize: 2
      args: -dim 1

   test:
      suffix: 2
      nsize: 3
      args: -dim 1 -stag_boundary_type_x none
      output_file: output/ex53_1.out

   test:
      suffix: 3
      nsize: 4
      args: -dim 1 -stag_boundary_type_x periodic
      output_file: output/ex53_1.out

   test:
      suffix: 4
      nsize: 4
      args: -dim 2
      output_file: output/ex53_1.out

   test:
      suffix: 5
      nsize: 4
      args: -dim 2 -stag_boundary_type_x none -stag_stencil_type star
      output_file: output/ex53_1.out

   test:
      suffix: 6
      nsize: 6
      args: -dim 2 -stag_boundary_type_y periodic -stag_stencil_width 2 -stag_dof_0 0 -stag_dof_1 1 -stag_dof_2 0
      output_file: output/ex53_1.out

   test:
      suffix: 7
      nsize: 8
      args: -dim 3
      output_file: output/ex53_1.out

   test:
      suffix: 8
      nsize: 8
      args: -dim 3 -stag_boundary_type_x none -stag_boundary_type_y periodic
      output_file: output/ex53_1.out

   test:
      suffix: 9
      nsize: 12
      args: -dim 3 -stag_boundary_type_x none -stag_boundary_type_y none -stag_boundary_type_z none -stag_stencil_type star -stag_dof_0 0 -stag_dof_1 0 -stag_dof_2 0 -stag_dof_3 1
      output_file: output/ex53_1.out

TEST*/
