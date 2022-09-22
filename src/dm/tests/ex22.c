
static char help[] = "Tests MatSetValuesBlockedStencil() in 3d.\n\n";

#include <petscmat.h>
#include <petscdm.h>
#include <petscdmda.h>

int main(int argc, char **argv)
{
  PetscInt        M = 3, N = 4, P = 2, s = 1, w = 2, i, m = PETSC_DECIDE, n = PETSC_DECIDE, p = PETSC_DECIDE;
  DM              da;
  Mat             mat;
  DMDAStencilType stencil_type = DMDA_STENCIL_BOX;
  PetscBool       flg          = PETSC_FALSE;
  MatStencil      idx[2], idy[2];
  PetscScalar    *values;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-P", &P, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-p", &p, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-s", &s, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-w", &w, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-star", &flg, NULL));
  if (flg) stencil_type = DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, stencil_type, M, N, P, m, n, p, w, s, 0, 0, 0, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMSetMatType(da, MATMPIBAIJ));
  PetscCall(DMCreateMatrix(da, &mat));

  idx[0].i = 1;
  idx[0].j = 1;
  idx[0].k = 0;
  idx[1].i = 2;
  idx[1].j = 1;
  idx[1].k = 0;
  idy[0].i = 1;
  idy[0].j = 2;
  idy[0].k = 0;
  idy[1].i = 2;
  idy[1].j = 2;
  idy[1].k = 0;
  PetscCall(PetscMalloc1(2 * 2 * w * w, &values));
  for (i = 0; i < 2 * 2 * w * w; i++) values[i] = i;
  PetscCall(MatSetValuesBlockedStencil(mat, 2, idx, 2, idy, values, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));

  /* Free memory */
  PetscCall(PetscFree(values));
  PetscCall(MatDestroy(&mat));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}
