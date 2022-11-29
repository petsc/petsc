
static char help[] = "Tests mirror boundary conditions in 3-d.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc, char **argv)
{
  PetscInt        M = 2, N = 3, P = 4, stencil_width = 1, dof = 1, m, n, p, xstart, ystart, zstart, i, j, k, c;
  DM              da;
  Vec             global, local;
  PetscScalar ****vglobal;
  PetscViewer     sview;
  PetscScalar     sum;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, 0, "-stencil_width", &stencil_width, 0));
  PetscCall(PetscOptionsGetInt(NULL, 0, "-dof", &dof, 0));

  PetscCall(DMDACreate3d(PETSC_COMM_WORLD, DM_BOUNDARY_MIRROR, DM_BOUNDARY_MIRROR, DM_BOUNDARY_MIRROR, DMDA_STENCIL_STAR, M, N, P, PETSC_DECIDE, PETSC_DECIDE, PETSC_DECIDE, dof, stencil_width, NULL, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDAGetCorners(da, &xstart, &ystart, &zstart, &m, &n, &p));

  PetscCall(DMCreateGlobalVector(da, &global));
  PetscCall(DMDAVecGetArrayDOF(da, global, &vglobal));
  for (k = zstart; k < zstart + p; k++) {
    for (j = ystart; j < ystart + n; j++) {
      for (i = xstart; i < xstart + m; i++) {
        for (c = 0; c < dof; c++) vglobal[k][j][i][c] = 1000 * k + 100 * j + 10 * i + c;
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(da, global, &vglobal));

  PetscCall(DMCreateLocalVector(da, &local));
  PetscCall(DMGlobalToLocalBegin(da, global, ADD_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(da, global, ADD_VALUES, local));

  PetscCall(VecSum(local, &sum));
  PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "sum %g\n", (double)PetscRealPart(sum)));
  PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, stdout));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sview));
  PetscCall(VecView(local, sview));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sview));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(DMDestroy(&da));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
     suffix: 2
     nsize: 3

TEST*/
