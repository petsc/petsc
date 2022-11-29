
static char help[] = "Tests mirror boundary conditions in 2-d.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc, char **argv)
{
  PetscInt       M = 8, N = 8, stencil_width = 1, dof = 1, m, n, xstart, ystart, i, j, c;
  DM             da;
  Vec            global, local;
  PetscScalar ***vglobal;
  PetscViewer    sview;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, 0, "-stencil_width", &stencil_width, 0));
  PetscCall(PetscOptionsGetInt(NULL, 0, "-dof", &dof, 0));

  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_MIRROR, DM_BOUNDARY_MIRROR, DMDA_STENCIL_STAR, M, N, PETSC_DECIDE, PETSC_DECIDE, dof, stencil_width, NULL, NULL, &da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDAGetCorners(da, &xstart, &ystart, 0, &m, &n, 0));

  PetscCall(DMCreateGlobalVector(da, &global));
  PetscCall(DMDAVecGetArrayDOF(da, global, &vglobal));
  for (j = ystart; j < ystart + n; j++) {
    for (i = xstart; i < xstart + m; i++) {
      for (c = 0; c < dof; c++) vglobal[j][i][c] = 100 * j + 10 * (i + 1) + c;
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(da, global, &vglobal));

  PetscCall(DMCreateLocalVector(da, &local));
  PetscCall(DMGlobalToLocalBegin(da, global, INSERT_VALUES, local));
  PetscCall(DMGlobalToLocalEnd(da, global, INSERT_VALUES, local));

  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sview));
  PetscCall(VecView(local, sview));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD, PETSC_COMM_SELF, &sview));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(global, PETSC_VIEWER_STDOUT_WORLD));

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
      nsize: 4
      filter: grep -v "Vec Object"

TEST*/
