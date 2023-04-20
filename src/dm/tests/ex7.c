
static char help[] = "Tests DMLocalToLocalxxx() for DMDA.\n\n";

#include <petscdmda.h>

int main(int argc, char **argv)
{
  PetscMPIInt     rank;
  PetscInt        M = 8, dof = 1, stencil_width = 1, i, start, end, P = 5, N = 6, m = PETSC_DECIDE, n = PETSC_DECIDE, p = PETSC_DECIDE, pt = 0, st = 0;
  PetscBool       flg = PETSC_FALSE, flg2, flg3;
  DMBoundaryType  periodic;
  DMDAStencilType stencil_type;
  DM              da;
  Vec             local, global, local_copy;
  PetscScalar     value;
  PetscReal       norm, work;
  PetscViewer     viewer;
  char            filename[64];
  FILE           *file;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dof", &dof, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-stencil_width", &stencil_width, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-periodic", &pt, NULL));

  periodic = (DMBoundaryType)pt;

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-stencil_type", &st, NULL));

  stencil_type = (DMDAStencilType)st;

  PetscCall(PetscOptionsHasName(NULL, NULL, "-grid2d", &flg2));
  PetscCall(PetscOptionsHasName(NULL, NULL, "-grid3d", &flg3));
  if (flg2) {
    PetscCall(DMDACreate2d(PETSC_COMM_WORLD, periodic, periodic, stencil_type, M, N, m, n, dof, stencil_width, NULL, NULL, &da));
  } else if (flg3) {
    PetscCall(DMDACreate3d(PETSC_COMM_WORLD, periodic, periodic, periodic, stencil_type, M, N, P, m, n, p, dof, stencil_width, NULL, NULL, NULL, &da));
  } else {
    PetscCall(DMDACreate1d(PETSC_COMM_WORLD, periodic, M, dof, stencil_width, NULL, &da));
  }
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));

  PetscCall(DMCreateGlobalVector(da, &global));
  PetscCall(DMCreateLocalVector(da, &local));
  PetscCall(VecDuplicate(local, &local_copy));

  /* zero out vectors so that ghostpoints are zero */
  value = 0;
  PetscCall(VecSet(local, value));
  PetscCall(VecSet(local_copy, value));

  PetscCall(VecGetOwnershipRange(global, &start, &end));
  for (i = start; i < end; i++) {
    value = i + 1;
    PetscCall(VecSetValues(global, 1, &i, &value, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(global));
  PetscCall(VecAssemblyEnd(global));

  PetscCall(DMGlobalToLocalBegin(da, global, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(da, global, INSERT_VALUES, local));

  PetscCall(DMLocalToLocalBegin(da, local, INSERT_VALUES, local_copy));
  PetscCall(DMLocalToLocalEnd(da, local, INSERT_VALUES, local_copy));

  PetscCall(PetscOptionsGetBool(NULL, NULL, "-save", &flg, NULL));
  if (flg) {
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
    PetscCall(PetscSNPrintf(filename, PETSC_STATIC_ARRAY_LENGTH(filename), "local.%d", rank));
    PetscCall(PetscViewerASCIIOpen(PETSC_COMM_SELF, filename, &viewer));
    PetscCall(PetscViewerASCIIGetPointer(viewer, &file));
    PetscCall(VecView(local, viewer));
    fprintf(file, "Vector with correct ghost points\n");
    PetscCall(VecView(local_copy, viewer));
    PetscCall(PetscViewerDestroy(&viewer));
  }

  PetscCall(VecAXPY(local_copy, -1.0, local));
  PetscCall(VecNorm(local_copy, NORM_MAX, &work));
  PetscCall(MPIU_Allreduce(&work, &norm, 1, MPIU_REAL, MPIU_MAX, PETSC_COMM_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Norm of difference %g should be zero\n", (double)norm));

  PetscCall(VecDestroy(&local_copy));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 8
      args: -dof 3 -stencil_width 2 -M 50 -N 50 -periodic

   test:
      suffix: 2
      nsize: 8
      args: -dof 3 -stencil_width 2 -M 50 -N 50 -periodic -grid2d
      output_file: output/ex7_1.out

   test:
      suffix: 3
      nsize: 8
      args: -dof 3 -stencil_width 2 -M 50 -N 50 -periodic -grid3d
      output_file: output/ex7_1.out

TEST*/
